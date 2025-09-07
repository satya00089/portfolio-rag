# index_portfolio_fixed.py
# pip install pymongo openai python-dotenv
import os
import json
import time
import math
import hashlib
from typing import List
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# modern openai client
from openai import OpenAI

load_dotenv()

MONGO_URI = os.environ["MONGODB_URI"]
MONGO_DB = os.environ.get("MONGODB_DB", "resume_rag")
MONGO_COLL = os.environ.get("MONGODB_COLL", "chunks")
OPENAI_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))  # characters
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

if not OPENAI_KEY:
    raise SystemExit("OPENAI_API_KEY required in environment")

# --- MongoDB client (avoid name 'client' collision) ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
coll = db[MONGO_COLL]

# --- OpenAI client (distinct name) ---
openai_client = OpenAI(api_key=OPENAI_KEY)

# load portfolio JSON (adjust path)
with open("portfolio.json", "r", encoding="utf-8") as f:
    portfolio = json.load(f)


# ---------- helpers ----------
def chunk_text(s: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    s = s.strip()
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
        i = max(end - overlap, end)
    return parts


def collect_text_parts(data: dict) -> List[dict]:
    parts = []
    p = data.get("personal", {})
    if p.get("summary"):
        parts.append({"text": p["summary"], "meta": {"section": "personal.summary"}})
    if data.get("summary"):
        parts.append({"text": data["summary"], "meta": {"section": "summary"}})
    if data.get("highlights"):
        parts.append(
            {
                "text": "Highlights:\n" + "\n".join(data["highlights"]),
                "meta": {"section": "highlights"},
            }
        )
    for proj in data.get("projects", []):
        txt = f"{proj.get('title','')}\n{proj.get('description','')}\nTags: {', '.join(proj.get('tags', []))}"
        parts.append(
            {
                "text": txt,
                "meta": {
                    "section": "project",
                    "id": proj.get("id"),
                    "title": proj.get("title"),
                    "url": proj.get("href"),
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


# Robust embed_batch supporting new/old SDK shapes
def embed_batch(inputs: List[str]) -> List[List[float]]:
    """
    Returns list of embedding vectors aligned with inputs.
    Uses openai_client.embeddings.create(...)
    """
    # call embeddings
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=inputs)

    # resp may be an object with .data or a dict; handle both
    data = None
    if hasattr(resp, "data"):
        data = resp.data
    elif isinstance(resp, dict) and "data" in resp:
        data = resp["data"]
    else:
        raise RuntimeError(f"Unexpected embeddings response shape: {type(resp)}")

    embeddings = []
    for item in data:
        # item may be object with .embedding or dict with ['embedding']
        if hasattr(item, "embedding"):
            embeddings.append(item.embedding)
        elif isinstance(item, dict) and "embedding" in item:
            embeddings.append(item["embedding"])
        else:
            raise RuntimeError(f"Unexpected embedding item shape: {type(item)}")
    return embeddings


def upsert_chunks(chunks):
    ops = []
    for c in chunks:
        doc = {
            "id": c["id"],
            "text": c["text"],
            "meta": c.get("meta", {}),
            "embedding": c.get("embedding"),
            "indexedAt": c.get("indexedAt")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ops.append(UpdateOne({"id": doc["id"]}, {"$set": doc}, upsert=True))
    if ops:
        res = coll.bulk_write(ops)
        return res.bulk_api_result
    return None


# ---------- main ----------
def main():
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
        embeddings = embed_batch(inputs)
        for j, emb in enumerate(embeddings):
            batch[j]["embedding"] = emb
        upsert_chunks(batch)
        time.sleep(0.5)

    print("Indexing complete")


if __name__ == "__main__":
    main()
