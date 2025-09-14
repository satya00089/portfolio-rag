"""API server for RAG on portfolio/resume using FastAPI, OpenAI, and MongoDB."""
import os
import math
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI


# --- Environment / config ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "portfolio")
MONGODB_COLL = os.getenv("MONGODB_COLL", "portfolio_chunks")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "4"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI must be set in environment")

# --- Clients ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
coll = mongo_client[MONGODB_DB][MONGODB_COLL]

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
def cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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
    k = req.k or TOP_K
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    # 1) embed the query
    try:
        emb_resp = openai_client.embeddings.create(model=EMBED_MODEL, input=q)
        # support new client response shape
        emb = (
            emb_resp.data[0].embedding
            if hasattr(emb_resp, "data")
            else emb_resp["data"][0]["embedding"]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {str(e)}")

    # 2) try Atlas vector search ($search / knnBeta). fallback to scanning & cosine.
    results = []
    try:
        # prefer $vectorSearch/$search with knnBeta — choose syntax available in your Atlas
        pipeline = [
            {"$search": {"index": "index", "knnBeta": {"vector": emb, "path": "embedding", "k": k}}},
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "text": 1,
                    "meta": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
            {"$limit": k},
        ]
        cursor = coll.aggregate(pipeline)
        results = list(cursor)
    except Exception:
        # fallback: compute cosine similarity across documents (fine for small dataset)
        docs = list(coll.find({}, {"id": 1, "text": 1, "meta": 1, "embedding": 1}))
        scored = []
        for d in docs:
            score = cosine_sim(emb, d.get("embedding") or [])
            scored.append(
                {
                    "id": d.get("id"),
                    "text": d.get("text"),
                    "meta": d.get("meta"),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:k]

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
        chat_resp = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, max_tokens=600, temperature=0.0
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
