"""Index the checked-in portfolio data with Bedrock Titan embeddings."""
import os
import json
import time
import math
import hashlib
import sys
from pathlib import Path
from typing import List
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.bedrock import EMBED_DIMENSIONS, EMBED_MODEL, create_runtime, embed_text

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))  # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
INDEX_DELAY = float(os.environ.get("BEDROCK_INDEX_DELAY", "0.5"))


# ---------- helpers ----------
def chunk_text(s: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
    s = s.strip()
    if not s:
        return []
    if len(s) <= chunk_size:
        return [s]
    parts = []
    i = 0
    n = len(s)
    while i < n:
        end = i + chunk_size
        if end < n:
            last_space = s.rfind(" ", i, end)
            if last_space > i + (chunk_size // 2):
                end = last_space
        parts.append(s[i:end].strip())
        if end >= n:
            break
        i = max(i + 1, end - overlap)
    return parts


def collect_text_parts(data: dict) -> List[dict]:
    parts = []
    p = data.get("personal", {})
    if p.get("summary"):
        parts.append({"text": p["summary"], "meta": {"section": "personal.summary"}})
    if p.get("contact"):
        contact_lines = []
        for k, v in p["contact"].items():
            if k == "socials":
                for social in v:
                    contact_lines.append(
                        f"{social.get('label', 'social')}: {social.get('url','')}"
                    )
            else:
                contact_lines.append(f"{k}: {v}")
        parts.append(
            {
                "text": "Contact:\n" + "\n".join(contact_lines),
                "meta": {"section": "personal.contact"},
            }
        )
    if data.get("summary"):
        parts.append({"text": data["summary"], "meta": {"section": "summary"}})
    if data.get("highlights"):
        parts.append(
            {
                "text": "Highlights:\n" + "\n".join(data["highlights"]),
                "meta": {"section": "highlights"},
            }
        )
    if data.get("projects", None):
        projects = [
            f"Project: {p.get('title','')}\n{p.get('description','')}\nTags: {', '.join(p.get('tags', []))}\n"
            for p in data.get("projects", [])
        ]
        projects_text = "\n ".join(projects)
        txt = f"Projects:\n {projects_text}"
        parts.append(
            {
                "text": txt,
                "meta": {"section": "projects"},
            }
        )

    for proj in data.get("projects", []):
        links = [
            f"label: {l.get('label')}, url: {l.get('url')}"
            for l in proj.get("links", [])
        ]
        txt = f"Project: {proj.get('title','')}\n{proj.get('description','')}\nTags: {', '.join(proj.get('tags', []))}\nLinks: {links}"
        parts.append(
            {
                "text": txt,
                "meta": {
                    "section": "project",
                    "id": proj.get("id"),
                    "title": proj.get("title"),
                    "url": proj.get("href"),
                    "links": proj.get("links", []),
                },
            }
        )
    for exp in data.get("experience", []):
        bullets = "\n".join(exp.get("bullets", []))
        txt = f"{exp.get('title','')} @ {exp.get('company','')}\n{exp.get('summary','')}\n{bullets}"
        parts.append(
            {
                "text": txt,
                "meta": {
                    "section": "experience",
                    "id": exp.get("id"),
                    "company": exp.get("company"),
                },
            }
        )
    for g in data.get("skills", []):
        skills_text = "\n".join(
            [
                f"{s.get('name')} — {s.get('years','')} yrs — {s.get('note','') or ''}"
                for s in g.get("skills", [])
            ]
        )
        parts.append(
            {
                "text": f"{g.get('title','Skills')}\n{skills_text}",
                "meta": {"section": "skills", "group": g.get("title")},
            }
        )
    if data.get("education"):
        parts.append(
            {
                "text": "Education:\n"
                + "\n".join(
                    [
                        f"{e.get('degree')} — {e.get('school')} ({e.get('date')})"
                        for e in data["education"]
                    ]
                ),
                "meta": {"section": "education"},
            }
        )
    if data.get("certifications"):
        parts.append(
            {
                "text": "Certifications:\n"
                + "\n".join(
                    [
                        f"{c.get('name')} — {c.get('issuer')} ({c.get('date')})"
                        for c in data["certifications"]
                    ]
                ),
                "meta": {"section": "certifications"},
            }
        )
    if data.get("extras"):
        parts.append(
            {
                "text": "Extras:\n" + json.dumps(data.get("extras")),
                "meta": {"section": "extras"},
            }
        )
    return parts


def text_to_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_batch(client, inputs: List[str]) -> List[List[float]]:
    """Titan accepts one input per request; keep the vectors in input order."""
    embeddings = []
    for text in inputs:
        embeddings.append(embed_text(client, text))
        if INDEX_DELAY:
            time.sleep(INDEX_DELAY)
    return embeddings


def upsert_chunks(collection, chunks):
    ops = []
    for c in chunks:
        doc = {
            "id": c["id"],
            "text": c["text"],
            "meta": c.get("meta", {}),
            "embedding": c.get("embedding"),
            "embedding_model": EMBED_MODEL,
            "embedding_dimensions": EMBED_DIMENSIONS,
            "indexedAt": c.get("indexedAt")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ops.append(UpdateOne({"id": doc["id"]}, {"$set": doc}, upsert=True))
    if ops:
        res = collection.bulk_write(ops)
        return res.bulk_api_result
    return None


# ---------- main ----------
def main():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise SystemExit("MONGODB_URI is required")
    source_name = os.getenv("BEDROCK_SOURCE_COLLECTION", "portfolio_chunks")
    target_name = os.getenv("BEDROCK_TARGET_COLLECTION", "portfolio_chunks_titan_v2")
    if target_name == source_name:
        raise SystemExit("Choose a different Bedrock target collection from your existing source")
    if BATCH_SIZE < 1 or INDEX_DELAY < 0:
        raise SystemExit("BATCH_SIZE must be positive and BEDROCK_INDEX_DELAY must be non-negative")
    coll = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)[os.getenv("MONGODB_DB", "portfolio")][target_name]
    mismatch = {"$or": [
        {"embedding_model": {"$ne": EMBED_MODEL}},
        {"embedding_dimensions": {"$ne": EMBED_DIMENSIONS}},
    ]}
    if coll.count_documents(mismatch):
        raise SystemExit("Target contains incompatible embeddings; choose a new collection")
    bedrock_client = create_runtime()
    with Path(__file__).with_name("portfolio.json").open("r", encoding="utf-8") as f:
        portfolio = json.load(f)
    parts = collect_text_parts(portfolio)
    docs = []
    for p in parts:
        subchunks = chunk_text(p["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, sc in enumerate(subchunks):
            chunk_meta = dict(p.get("meta", {}))
            chunk_meta.update({"part_index": i})
            _id = text_to_id(sc + json.dumps(chunk_meta, sort_keys=True))
            docs.append({"id": _id, "text": sc, "meta": chunk_meta})
    print(f"Prepared {len(docs)} chunks")

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        inputs = [d["text"] for d in batch]
        print(
            f"Embedding batch {i//BATCH_SIZE + 1}/{math.ceil(len(docs)/BATCH_SIZE)} size {len(inputs)}"
        )
        embeddings = embed_batch(bedrock_client, inputs)
        for j, emb in enumerate(embeddings):
            batch[j]["embedding"] = emb
        upsert_chunks(coll, batch)

    print("Indexing complete")


if __name__ == "__main__":
    main()
