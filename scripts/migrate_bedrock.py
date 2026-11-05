"""Copy existing portfolio chunks into a separate Titan-embedded collection."""
import argparse
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.bedrock import EMBED_DIMENSIONS, EMBED_MODEL, create_runtime, embed_text


def migrate(source, target, source_name, runtime, resume=False, delay=0.5):
    """Write only to an empty target, or resume a previous matching migration."""
    if source_name == target.name:
        raise ValueError("Source and target collections must be different")

    documents = list(source.find({}, {"id": 1, "text": 1, "meta": 1}))
    if not documents:
        raise ValueError("Source collection is empty; nothing was migrated")
    if any(not isinstance(doc.get("text"), str) or not doc["text"].strip() for doc in documents):
        raise ValueError("Every source document must contain non-empty text")

    if target.count_documents({}):
        if not resume:
            raise ValueError("Target is not empty; choose an empty target or use --resume")
        mismatch = {"$or": [
            {"migration_source": {"$ne": source_name}},
            {"embedding_model": {"$ne": EMBED_MODEL}},
            {"embedding_dimensions": {"$ne": EMBED_DIMENSIONS}},
        ]}
        if target.count_documents(mismatch):
            raise ValueError("Target contains documents from another migration or embedding model")

    for index, doc in enumerate(documents, start=1):
        prior = target.find_one({"_id": doc["_id"]}) if resume else None
        if prior and prior.get("text") == doc["text"] and prior.get("meta") == (doc.get("meta") or {}):
            print(f"[{index}/{len(documents)}] Already migrated")
            continue
        vector = embed_text(runtime, doc["text"])
        target.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "id": str(doc.get("id") or doc["_id"]),
                "text": doc["text"],
                "meta": doc.get("meta") or {},
                "embedding": vector,
                "embedding_model": EMBED_MODEL,
                "embedding_dimensions": EMBED_DIMENSIONS,
                "migration_source": source_name,
            }},
            upsert=True,
        )
        print(f"[{index}/{len(documents)}] Migrated")
        if delay and index < len(documents):
            time.sleep(delay)

    actual = target.count_documents({"migration_source": source_name})
    if actual != len(documents):
        raise RuntimeError(f"Target has {actual} chunks; expected {len(documents)}. Review before switching.")
    print(f"Migration complete: {actual} chunks in {target.name}. The source was not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="portfolio_chunks", help="Existing OpenAI-embedded collection")
    parser.add_argument("--target", default="portfolio_chunks_titan_v2", help="New, separate collection")
    parser.add_argument("--resume", action="store_true", help="Resume a previous migration to the same target")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between embedding requests")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("--source and --target must be different")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    uri = os.getenv("MONGODB_URI")
    if not uri:
        parser.error("MONGODB_URI is required")
    db = MongoClient(uri, serverSelectionTimeoutMS=5000)[os.getenv("MONGODB_DB", "portfolio")]
    migrate(db[args.source], db[args.target], args.source, create_runtime(), args.resume, args.delay)


if __name__ == "__main__":
    main()
