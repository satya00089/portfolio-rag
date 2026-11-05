# Portfolio assistant on Amazon Bedrock

This FastAPI service answers portfolio questions for the existing CLI. It uses
Amazon Titan Text Embeddings V2 to embed each question, compares that vector
with portfolio chunks in MongoDB, and asks GPT-6 Luna on Amazon Bedrock to
answer from the selected chunks. The existing `POST /api/query` request and
`{ "answer": ..., "sources": [...] }` response stay the same.

## Models, Region, and cost

The defaults are designed for a small, text-only portfolio:

| Purpose | Default |
| --- | --- |
| Bedrock source Region | `ap-south-1` (Mumbai) |
| Answer model | `global.openai.gpt-6-luna` |
| Reasoning effort | `low` |
| Embeddings | `amazon.titan-embed-text-v2:0`, 1,024 dimensions |
| MongoDB collection | `portfolio_chunks_titan_v2` |

GPT-6 Luna's global inference profile can route requests outside India. The
source Region is Mumbai; it is **not** a data-residency guarantee. Model
availability, quotas and pricing depend on the AWS account and Region. Bedrock
on-demand inference is billed through AWS for usage; existing AWS promotional
credits, if any, may have their own expiration rules. Check the current
[Bedrock model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-luna.html)
and [pricing](https://aws.amazon.com/bedrock/pricing/).

The default uses a direct cosine-similarity scan of the configured MongoDB
collection. This keeps a small portfolio simple and does not require an Atlas
vector index. If the collection becomes large, switch to a properly indexed
vector search rather than scanning every chunk.

## Configuration

Configure these as environment variables in Vercel for the API, and locally
when running the migration. Do not put secrets in source control or send them
in chat. Boto3 1.39.0 or later reads a Bedrock API key from
`AWS_BEARER_TOKEN_BEDROCK`; alternatively it can use normal AWS credentials.

| Variable | Value |
| --- | --- |
| `AWS_BEARER_TOKEN_BEDROCK` | Your Bedrock API key, stored as a secret. |
| `BEDROCK_REGION` | `ap-south-1` by default. |
| `BEDROCK_CHAT_MODEL_ID` | `global.openai.gpt-6-luna` by default. |
| `BEDROCK_REASONING_EFFORT` | `low` by default, for GPT-6 models. |
| `BEDROCK_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` by default. |
| `BEDROCK_EMBEDDING_DIMENSIONS` | `1024` by default; Titan also supports `512` and `256`. |
| `MONGODB_URI` | Your existing MongoDB connection string, stored as a secret. |
| `MONGODB_DB` | Existing database; defaults to `portfolio`. |
| `MONGODB_COLL` | API collection; set to the new target, e.g. `portfolio_chunks_titan_v2`. |
| `CORS_ALLOW_ORIGINS` | Comma-separated browser origins, e.g. `https://satya00089.github.io`. |
| `TOP_K` | Number of context chunks to return; defaults to `4` and is capped at `10`. |

Bedrock API keys also expire: short-term keys last no longer than 12 hours,
and long-term keys expire after the period chosen when they are created. For a
long-running deployment, use a suitable long-term key and rotate it before
expiry, or use refreshable AWS credentials. Key expiry is separate from AWS
billing credits. Ensure the
key's IAM permissions and model access allow invoking both Titan Text
Embeddings V2 and the GPT-6 Luna global inference profile in the configured
Region. AWS may require permissions for the profile, its destination models,
and the default Bedrock project. See [AWS API key setup](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)
and the [GPT-6 Luna model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-luna.html).

## Migrate existing portfolio data safely

OpenAI and Titan vectors are different, even if configured with the same
number of dimensions. Do **not** compare or overwrite the old embeddings. The
migration reads existing chunk text and metadata, calls Bedrock once per chunk,
and writes new vectors to a separate collection. It never writes to the source.
It does incur Bedrock usage charges.

1. In Vercel, note the existing `MONGODB_DB` and `MONGODB_COLL` values. The
   source may have a different name from the example below. Do not change the
   production collection or deploy the new API yet.
2. Create a local virtual environment and install the dependencies:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set `MONGODB_URI`, `MONGODB_DB`, `AWS_BEARER_TOKEN_BEDROCK`, and, if
   needed, the Bedrock model, Region and dimension variables locally. Use the
   **same** embedding model and dimensions that the API will use.
4. Pick a new empty collection and run the migration, replacing the source name
   with the current Vercel `MONGODB_COLL` value:

   ```sh
   python -m scripts.migrate_bedrock --source portfolio_chunks --target portfolio_chunks_titan_v2
   ```

   The command refuses to use the same source and target, or to overwrite a
   nonempty target. If it stops partway through, rerun with `--resume` and the
   same source, target, model and dimensions. It checks for incompatible data
   before resuming. Confirm the reported count matches the source collection.
5. Set the new Vercel environment variables, including `MONGODB_COLL` set to
   the new target. Deploy the new code and send a test question. Keep the old
   collection intact until you have verified the new deployment.

If the API says portfolio information is not indexed, check the target
collection, `MONGODB_DB`, embedding model and dimensions. The API only reads
chunks tagged with the exact configured embedding model and dimension count;
this prevents old OpenAI vectors from being mixed in. `/api/health` is a
liveness check, not a check of Bedrock credentials or MongoDB connectivity.

## Update portfolio information later

For a fresh index of the checked-in `scripts/portfolio.json`, run:

```sh
python -m scripts.index_portfolio
```

The script defaults to the separate `portfolio_chunks_titan_v2` collection.
Set `BEDROCK_TARGET_COLLECTION` if the intended destination is different and
`BEDROCK_SOURCE_COLLECTION` if your old collection is not `portfolio_chunks`;
the indexer refuses to write to that source or to a collection containing
incompatible embeddings. `BEDROCK_INDEX_DELAY` controls the delay in seconds
between Titan requests (default `0.5`). The
checked-in JSON may be older than production data: review it before indexing.
The indexer upserts chunks and does not delete stale chunks automatically.
For a clean rebuild, use a new empty collection, verify it, then update the
API's `MONGODB_COLL` setting. Changing the embedding model or dimensions also
requires a fresh reindex.

The API is publicly reachable. CORS controls which browsers may read responses;
it is **not** authentication or a rate limit. Set AWS budget alerts and use a
real rate-limiting layer such as a Vercel firewall rule if you need to bound
cost from unwanted requests. Never put the Bedrock key in the frontend.

`scripts/rag_function.js` is a legacy MongoDB Atlas function using OpenAI's API;
it is not used by this Vercel deployment and has not been migrated to Bedrock.
