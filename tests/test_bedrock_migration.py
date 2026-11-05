"""Offline contract checks for the Bedrock API and safe migration."""
import io
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api import bedrock, main
from scripts import index_portfolio, migrate_bedrock


class FakeRuntime:
    def __init__(self):
        self.embedding_calls = []
        self.chat_calls = []

    def invoke_model(self, **kwargs):
        self.embedding_calls.append(kwargs)
        vector = [1.0] + [0.0] * (bedrock.EMBED_DIMENSIONS - 1)
        return {"body": io.BytesIO(json.dumps({"embedding": vector}).encode())}

    def converse(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": "A grounded answer."}]}}}


class QueryCollection:
    def __init__(self, docs):
        self.docs = docs
        self.last_filter = None

    def find(self, query, projection):
        self.last_filter = query
        return [doc for doc in self.docs if all(doc.get(key) == value for key, value in query.items())]


class MigrationCollection:
    def __init__(self, name, docs=()):
        self.name = name
        self.docs = {doc["_id"]: dict(doc) for doc in docs}

    def find(self, query, projection):
        return [dict(doc) for doc in self.docs.values()]

    def find_one(self, query):
        return self.docs.get(query["_id"])

    def count_documents(self, query):
        if "$or" in query:
            return sum(any(doc.get(k) != v["$ne"] for condition in query["$or"] for k, v in condition.items()) for doc in self.docs.values())
        return sum(all(doc.get(k) == v for k, v in query.items()) for doc in self.docs.values())

    def update_one(self, query, update, upsert):
        self.docs[query["_id"]] = {"_id": query["_id"], **update["$set"]}


class BedrockTests(unittest.TestCase):
    def test_inspection_reports_legacy_dimensions_without_writing(self):
        source = MigrationCollection("portfolio_chunks", [
            {"_id": "a", "text": "A fact", "embedding": [0.1] * 1536},
            {"_id": "b", "text": "Another fact", "embedding": [0.2] * 1536},
        ])
        before = {key: dict(value) for key, value in source.docs.items()}
        report = migrate_bedrock.inspect_source(source)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["dimensions"], {1536: 2})
        self.assertEqual(report["models"], {"not recorded": 2})
        self.assertEqual(report["missing_text"], 0)
        self.assertEqual(source.docs, before)

    def test_indexing_chunks_overlap_and_reject_invalid_settings(self):
        chunks = index_portfolio.chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)
        self.assertEqual(chunks, ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"])
        with self.assertRaises(ValueError):
            index_portfolio.chunk_text("text", chunk_size=4, overlap=4)

    def test_titan_and_luna_requests(self):
        runtime = FakeRuntime()
        vector = bedrock.embed_text(runtime, "A project")
        self.assertEqual(len(vector), bedrock.EMBED_DIMENSIONS)
        request = runtime.embedding_calls[0]
        self.assertEqual(request["modelId"], bedrock.EMBED_MODEL)
        self.assertEqual(json.loads(request["body"]), {
            "inputText": "A project", "dimensions": bedrock.EMBED_DIMENSIONS, "normalize": True,
        })
        answer = bedrock.answer_question(runtime, "Be factual", "Portfolio facts", "Who?")
        self.assertEqual(answer, "A grounded answer.")
        self.assertEqual(runtime.chat_calls[0]["modelId"], bedrock.CHAT_MODEL)
        self.assertEqual(runtime.chat_calls[0]["messages"][0]["content"][0]["text"], "Who?")

    def test_rejects_embedding_with_wrong_dimensions(self):
        runtime = FakeRuntime()
        with patch.object(runtime, "invoke_model", return_value={"body": io.BytesIO(b'{"embedding":[1]}')}):
            with self.assertRaises(ValueError):
                bedrock.embed_text(runtime, "A project")

    def test_query_uses_only_compatible_vectors_and_keeps_response_contract(self):
        vector = [1.0] + [0.0] * (bedrock.EMBED_DIMENSIONS - 1)
        collection = QueryCollection([
            {"_id": "new", "id": "new", "text": "Bedrock facts", "meta": {}, "embedding": vector,
             "embedding_model": bedrock.EMBED_MODEL, "embedding_dimensions": bedrock.EMBED_DIMENSIONS},
            {"_id": "old", "id": "old", "text": "OpenAI facts", "embedding": vector},
        ])
        runtime = FakeRuntime()
        with patch.object(main, "get_collection", return_value=collection), patch.object(main, "get_runtime", return_value=runtime):
            response = main.query_rag(main.QueryRequest(q="What did Satya build?"))
        self.assertEqual(response["answer"], "A grounded answer.")
        self.assertEqual([item["id"] for item in response["sources"]], ["new"])
        self.assertEqual(collection.last_filter, {"embedding_model": bedrock.EMBED_MODEL, "embedding_dimensions": bedrock.EMBED_DIMENSIONS})
        self.assertNotIn("OpenAI facts", runtime.chat_calls[0]["system"][1]["text"])

    def test_empty_collection_and_provider_failure_are_safe(self):
        runtime = FakeRuntime()
        with patch.object(main, "get_runtime", return_value=runtime), patch.object(main, "get_collection", return_value=QueryCollection([])):
            with self.assertRaises(HTTPException) as error:
                main.query_rag(main.QueryRequest(q="Hello"))
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(runtime.chat_calls, [])
        self.assertEqual(runtime.embedding_calls, [])
        valid = {"_id": "a", "text": "A fact", "embedding": [1.0] * bedrock.EMBED_DIMENSIONS,
                 "embedding_model": bedrock.EMBED_MODEL, "embedding_dimensions": bedrock.EMBED_DIMENSIONS}
        with patch.object(main, "get_collection", return_value=QueryCollection([valid])), patch.object(main, "get_runtime", side_effect=RuntimeError("sensitive provider detail")):
            with self.assertRaises(HTTPException) as error:
                main.query_rag(main.QueryRequest(q="Hello"))
        self.assertEqual(error.exception.status_code, 503)
        self.assertNotIn("sensitive provider detail", error.exception.detail)

    def test_migration_is_separate_and_resume_does_not_reembed(self):
        source = MigrationCollection("portfolio_chunks", [{"_id": "a", "id": "a", "text": "A fact", "meta": {"section": "about"}}])
        target = MigrationCollection("portfolio_chunks_titan_v2")
        runtime = FakeRuntime()
        original = dict(source.docs["a"])
        migrate_bedrock.migrate(source, target, source.name, runtime, delay=0)
        self.assertEqual(source.docs["a"], original)
        self.assertEqual(target.docs["a"]["embedding_model"], bedrock.EMBED_MODEL)
        self.assertEqual(target.docs["a"]["embedding_dimensions"], bedrock.EMBED_DIMENSIONS)
        with self.assertRaises(ValueError):
            migrate_bedrock.migrate(source, target, source.name, runtime, delay=0)
        migrate_bedrock.migrate(source, target, source.name, runtime, resume=True, delay=0)
        self.assertEqual(len(runtime.embedding_calls), 1)
        with self.assertRaises(ValueError):
            migrate_bedrock.migrate(source, source, source.name, runtime, delay=0)

    def test_resume_refuses_unrelated_target_documents(self):
        source = MigrationCollection("portfolio_chunks", [{"_id": "a", "text": "A fact"}])
        target = MigrationCollection("new", [{"_id": "other", "text": "Do not overwrite"}])
        with self.assertRaises(ValueError):
            migrate_bedrock.migrate(source, target, source.name, FakeRuntime(), resume=True, delay=0)
        self.assertEqual(target.docs["other"]["text"], "Do not overwrite")


if __name__ == "__main__":
    unittest.main()
