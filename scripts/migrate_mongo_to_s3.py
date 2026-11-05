"""Copy MongoDB text to S3 Vectors using Titan, leaving MongoDB untouched."""
import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import titan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Generate vectors and write to the configured S3 index")
    args = parser.parse_args()
    load_dotenv()
    uri = os.environ["MONGODB_URI"]
    db = os.getenv("MONGODB_DB", "portfolio")
    collection = os.getenv("MONGODB_COLL", "portfolio_chunks")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    try:
        docs = list(client[db][collection].find({}, {"id": 1, "text": 1, "meta": 1}))
    finally:
        client.close()
    if not docs:
        raise SystemExit("No source documents found; check the MongoDB database and collection.")
    keys = [str(doc.get("id") or doc["_id"]) for doc in docs]
    if len(set(keys)) != len(keys):
        raise SystemExit("Source documents have duplicate IDs; no vectors were written.")
    if any(not isinstance(doc.get("text"), str) or not doc["text"].strip() for doc in docs):
        raise SystemExit("A source document has no text; no vectors were written.")
    print(f"Found {len(docs)} source documents in {db}.{collection}.")
    if not args.apply:
        print("Dry run only; no Bedrock requests or S3 writes. Use --apply to copy.")
        return
    bucket = os.environ["S3_VECTOR_BUCKET"]
    index = os.getenv("S3_VECTOR_INDEX", "portfolio-titan-v2")
    vectors = boto3.client("s3vectors", region_name=os.getenv("BEDROCK_REGION", "ap-south-1"))
    for i, (key, doc) in enumerate(zip(keys, docs), 1):
        vector = titan.embed(doc["text"])
        vectors.put_vectors(vectorBucketName=bucket, indexName=index, vectors=[{
            "key": key, "data": {"float32": vector},
            "metadata": {"text": doc["text"], "meta_json": json.dumps(doc.get("meta") or {}, default=str)},
        }])
        print(f"Copied {i}/{len(docs)}")
    print("Copy complete. MongoDB data was not modified.")


if __name__ == "__main__":
    main()
