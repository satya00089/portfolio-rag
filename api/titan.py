"""Small Bedrock client shared by indexing, migration and query code."""
import json
import math
import os
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

MODEL = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


def embed(text: str) -> list[float]:
    """Create a Titan vector using a Bedrock API key, without logging secrets."""
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK must be set")
    region = os.getenv("BEDROCK_REGION", "ap-south-1")
    if not region or not all(c.isalnum() or c == "-" for c in region):
        raise ValueError("Invalid BEDROCK_REGION")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot embed empty text")
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(MODEL, safe='')}/invoke"
    request = Request(
        url,
        data=json.dumps({"inputText": text, "dimensions": DIMENSIONS, "normalize": True}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            vector = json.load(response).get("embedding")
    except HTTPError as exc:
        # Do not propagate request headers or response bodies which may contain sensitive data.
        raise RuntimeError(f"Bedrock embedding request failed (HTTP {exc.code})") from exc
    if (not isinstance(vector, list) or len(vector) != DIMENSIONS
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
            or not any(vector)):
        raise RuntimeError("Bedrock returned an invalid Titan embedding")
    return vector
