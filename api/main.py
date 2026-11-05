"""API server for RAG on portfolio/resume using Bedrock and S3 Vectors."""
import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from openai import OpenAI
from api import titan


# --- Environment / config ---
S3_VECTOR_BUCKET = os.getenv("S3_VECTOR_BUCKET")
S3_VECTOR_INDEX = os.getenv("S3_VECTOR_INDEX", "portfolio-titan-v2")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
BEDROCK_API_KEY = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL_ID", "global.openai.gpt-6-luna")
TOP_K = int(os.getenv("TOP_K", "4"))

if not S3_VECTOR_BUCKET:
    raise RuntimeError("S3_VECTOR_BUCKET must be set in environment")
if not BEDROCK_API_KEY:
    raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK must be set in environment")

# --- Clients ---
bedrock_client = OpenAI(
    api_key=BEDROCK_API_KEY,
    base_url=f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com/openai/v1",
)
vector_client = boto3.client("s3vectors", region_name=BEDROCK_REGION)

# --- FastAPI app ---
app = FastAPI(title="RAG API for Portfolio")

# CORS: tighten this in production
allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
if allow_origins == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allow_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response models ---
class QueryRequest(BaseModel):
    """Request model for /api/query."""
    q: str
    k: Optional[int] = None


class Source(BaseModel):
    """A source document returned from the vector DB."""
    id: str
    text: str
    meta: Optional[dict] = {}
    score: float


class QueryResponse(BaseModel):
    """Response model for /api/query."""
    answer: str
    sources: List[Source]


# --- Helpers ---
def normalize_result(r):
    """Normalize a DB result to Source model."""
    return {
        "id": r.get("id") or r.get("_id"),
        "text": r.get("text") or "",
        "meta": r.get("meta") or {},
        "score": float(r.get("score") or 0),
    }


@app.get('/')
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# --- Endpoint: health ---
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}




# --- Endpoint: query RAG ---
@app.post("/api/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Handle a query with RAG: embed, search, chat completion."""
    q = req.q.strip()
    k = TOP_K if req.k is None else req.k
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    if not 1 <= k <= 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")

    # 1) embed the query
    try:
        emb = titan.embed(q)
    except Exception as e:
        logging.exception("Bedrock embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service unavailable") from e

    # 2) Search the Titan vectors. S3 returns cosine distance, not similarity.
    try:
        response = vector_client.query_vectors(
            vectorBucketName=S3_VECTOR_BUCKET, indexName=S3_VECTOR_INDEX,
            queryVector={"float32": emb}, topK=k, returnMetadata=True, returnDistance=True,
        )
        if response.get("distanceMetric") != "cosine":
            raise ValueError("S3 vector index must use cosine distance")
        results = []
        for vector in response.get("vectors", []):
            metadata = vector.get("metadata") or {}
            results.append({
                "id": vector["key"], "text": metadata.get("text", ""),
                "meta": json.loads(metadata.get("meta_json", "{}")),
                "score": 1.0 - vector["distance"],
            })
    except Exception as e:
        logging.exception("S3 vector search failed")
        raise HTTPException(status_code=502, detail="Vector search unavailable") from e
    if not results:
        raise HTTPException(status_code=503, detail="No portfolio vectors found; check S3 index and migration")

    # normalize results
    sources = [normalize_result(r) for r in results]

    # 3) assemble context and call chat completion
    context_text = "\n\n---\n\n".join(
        [
            f"SOURCE {i+1} (score:{s['score']:.4f}):\n{s['text']}"
            for i, s in enumerate(sources)
        ]
    )
    system_prompt = (
        "You are a concise, factual assistant that answers questions about Satya's resume and portfolio. "
        "Use only the provided CONTEXT to form your answers; do not rely on external knowledge except for "
        "very brief clarifications. If the information is not present in the CONTEXT, respond with "
        "\"I don't know\" or a brief honest statement (e.g. \"I don't have that information in the provided context\"). "
        "Be concise and clear: prefer short paragraphs or bullet points and avoid speculation. "
        "Do not include raw source text longer than 200 characters; summarize and cite the source instead. "
        "If the user requests actionable changes (for example, resume edits), give step-by-step, prioritized suggestions. "
        "If the user's question is ambiguous, ask one clear clarifying question. "
        "Respect privacy: do not invent contact details, personal identifiers, or confidential data."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"CONTEXT:\n{context_text}"},
        {"role": "user", "content": q},
    ]

    try:
        chat_resp = bedrock_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, max_completion_tokens=600,
            reasoning_effort="none", temperature=0.0,
        )
        # support different response shapes
        if hasattr(chat_resp, "choices") and chat_resp.choices:
            answer = (
                chat_resp.choices[0].message.content
                if hasattr(chat_resp.choices[0], "message")
                else chat_resp.choices[0].text
            )
        else:
            # dict-like fallback
            answer = chat_resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat completion error: {str(e)}")

    return {"answer": answer, "sources": sources}
