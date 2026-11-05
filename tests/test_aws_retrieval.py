import asyncio
import importlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.stub import Stubber

from api import titan


def test_titan_uses_bedrock_key_and_expected_dimensions(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")
    monkeypatch.setenv("BEDROCK_REGION", "ap-south-1")
    response = io.BytesIO(json.dumps({"embedding": [0.5] * 1024}).encode())
    open_url = MagicMock(return_value=response)
    monkeypatch.setattr(titan, "urlopen", open_url)
    assert len(titan.embed("portfolio")) == 1024
    request = open_url.call_args.args[0]
    assert request.full_url == "https://bedrock-runtime.ap-south-1.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke"
    assert json.loads(request.data)["dimensions"] == 1024
    assert request.get_header("Authorization") == "Bearer test-key"


def test_titan_rejects_wrong_dimensions(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")
    monkeypatch.setattr(titan, "urlopen", lambda *a, **kw: io.BytesIO(b'{"embedding": [1]}'))
    with pytest.raises(RuntimeError, match="invalid Titan embedding"):
        titan.embed("portfolio")


def load_api(monkeypatch, vector_client):
    monkeypatch.setenv("S3_VECTOR_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: vector_client)
    sys.modules.pop("api.main", None)
    return importlib.import_module("api.main")


def test_query_uses_s3_sources_and_preserves_response(monkeypatch):
    vectors = MagicMock()
    vectors.query_vectors.return_value = {"distanceMetric": "cosine", "vectors": [{
        "key": "project-1", "distance": 0.2,
        "metadata": {"text": "Built Diagramwise", "meta_json": '{"section":"projects"}'},
    }]}
    api = load_api(monkeypatch, vectors)
    monkeypatch.setattr(titan, "embed", lambda text: [0.5] * 1024)
    api.bedrock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A project."))]))
    )))
    result = asyncio.run(api.query_rag(api.QueryRequest(q="What did you build?")))
    assert result["answer"] == "A project."
    assert result["sources"][0]["meta"] == {"section": "projects"}
    assert result["sources"][0]["score"] == pytest.approx(0.8)
    assert vectors.query_vectors.call_args.kwargs["queryVector"] == {"float32": [0.5] * 1024}
    assert "Built Diagramwise" in api.bedrock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]


def test_query_request_matches_s3_vectors_sdk(monkeypatch):
    client = boto3.client("s3vectors", region_name="ap-south-1", aws_access_key_id="test", aws_secret_access_key="test")
    api = load_api(monkeypatch, client)
    monkeypatch.setattr(titan, "embed", lambda text: [0.5] * 1024)
    api.bedrock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A project."))]))
    )))
    with Stubber(client) as stub:
        stub.add_response("query_vectors", {"distanceMetric": "cosine", "vectors": [{"key": "p1", "distance": 0.2, "metadata": {"text": "A project", "meta_json": "{}"}}]}, {
            "vectorBucketName": "test-bucket", "indexName": "portfolio-titan-v2", "queryVector": {"float32": [0.5] * 1024},
            "topK": 4, "returnMetadata": True, "returnDistance": True,
        })
        result = asyncio.run(api.query_rag(api.QueryRequest(q="What did you build?")))
    assert result["sources"][0]["id"] == "p1"


def test_missing_vectors_returns_error_instead_of_answer(monkeypatch):
    vectors = MagicMock()
    vectors.query_vectors.return_value = {"distanceMetric": "cosine", "vectors": []}
    api = load_api(monkeypatch, vectors)
    monkeypatch.setattr(titan, "embed", lambda text: [0.5] * 1024)
    with pytest.raises(api.HTTPException) as exc:
        asyncio.run(api.query_rag(api.QueryRequest(q="Hello")))
    assert exc.value.status_code == 503


def test_migration_dry_run_and_copy_preserve_mongo(monkeypatch):
    from scripts import migrate_mongo_to_s3 as migration

    docs = [{"_id": "db1", "id": "project-1", "text": "Built Diagramwise", "meta": {"section": "projects"}}]
    collection = MagicMock()
    collection.find.return_value = docs
    mongo = MagicMock()
    mongo.__getitem__.return_value.__getitem__.return_value = collection
    monkeypatch.setattr(migration, "MongoClient", lambda *a, **kw: mongo)
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setenv("S3_VECTOR_BUCKET", "test-bucket")
    vectors = MagicMock()
    monkeypatch.setattr(migration.boto3, "client", lambda *a, **kw: vectors)
    monkeypatch.setattr(titan, "embed", lambda text: [0.5] * 1024)

    monkeypatch.setattr(sys, "argv", ["migrate"])
    migration.main()
    vectors.put_vectors.assert_not_called()

    monkeypatch.setattr(sys, "argv", ["migrate", "--apply"])
    migration.main()
    vector = vectors.put_vectors.call_args.kwargs["vectors"][0]
    assert vector["key"] == "project-1"
    assert vector["metadata"]["text"] == docs[0]["text"]
    assert json.loads(vector["metadata"]["meta_json"]) == docs[0]["meta"]
    collection.update_one.assert_not_called()
    collection.delete_many.assert_not_called()
