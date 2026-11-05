"""Bedrock inference shared by the API and indexing tools."""
import json
import math
import os

import boto3
from botocore.config import Config

REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
EMBED_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIMENSIONS = int(os.getenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024"))
CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL_ID", "global.openai.gpt-6-luna")
REASONING_EFFORT = os.getenv("BEDROCK_REASONING_EFFORT", "low")

if EMBED_DIMENSIONS not in (256, 512, 1024):
    raise ValueError("BEDROCK_EMBEDDING_DIMENSIONS must be 256, 512, or 1024")


def create_runtime():
    """Use AWS_BEARER_TOKEN_BEDROCK or the standard AWS credential chain."""
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(connect_timeout=5, read_timeout=40, retries={"max_attempts": 2}),
    )


def embed_text(client, text):
    """Generate a Titan Text Embeddings V2 vector for one text."""
    result = client.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS, "normalize": True}),
    )
    vector = json.loads(result["body"].read())["embedding"]
    if (
        not isinstance(vector, list)
        or len(vector) != EMBED_DIMENSIONS
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
    ):
        raise ValueError("Bedrock returned an embedding with unexpected dimensions")
    return vector


def answer_question(client, system, context, question):
    """Generate an answer using a Bedrock Converse model."""
    args = {
        "modelId": CHAT_MODEL,
        "system": [{"text": system}, {"text": f"CONTEXT:\n{context}"}],
        "messages": [{"role": "user", "content": [{"text": question}]}],
        "inferenceConfig": {"maxTokens": 600},
    }
    # Other Bedrock model providers may use different reasoning parameters.
    if "openai.gpt-6" in CHAT_MODEL:
        args["additionalModelRequestFields"] = {"reasoning": {"effort": REASONING_EFFORT}}
    response = client.converse(**args)
    answer = "\n".join(
        block["text"] for block in response["output"]["message"]["content"] if block.get("text")
    ).strip()
    if not answer:
        raise ValueError("Bedrock returned no answer text")
    return answer
