# Move portfolio retrieval to Amazon S3 Vectors

The API uses Titan Text Embeddings V2 and S3 Vectors in `ap-south-1` by default.
GPT-6 Luna uses Bedrock's global inference profile, which can route inference
outside Mumbai. Its OpenAI-compatible client talks to Bedrock, not OpenAI.
The original MongoDB documents and OpenAI vectors are not changed by the copy.

## Before deploying

1. Install the dependencies from `requirements.txt`. Create a **new S3 vector
   bucket** in Mumbai and a vector index named `portfolio-titan-v2` (or choose
   your own name). Use `float32`, **1024 dimensions**, and **cosine** distance.
   Set `text` and `meta_json` as non-filterable metadata keys when creating
   the index; these store the original source text and metadata. These settings
   cannot be fixed by changing the application after the index is created.
2. Set `S3_VECTOR_BUCKET`, `S3_VECTOR_INDEX` if you used a different name,
   `BEDROCK_REGION=ap-south-1`, and `AWS_BEARER_TOKEN_BEDROCK` in a secure
   local environment. Provide normal AWS credentials for S3 Vectors through
   the standard AWS SDK credential chain (for example a restricted IAM role
   or access keys). A Bedrock API key alone does **not** authorize S3 requests.
   Never commit credentials or paste them into source control.
3. Give the migration identity permission for `s3vectors:PutVectors` on the
   index; the deployed API needs `s3vectors:QueryVectors` and
   `s3vectors:GetVectors`. Creating the bucket and index needs additional
   permissions. The Bedrock key's identity also needs permission to invoke
   Titan and the GPT-6 Luna inference profile.
4. Set `MONGODB_URI` and, if the deployed values are not the defaults,
   `MONGODB_DB` and `MONGODB_COLL` **locally for the migration only**. Defaults
   are `portfolio` and `portfolio_chunks`. Run a read-only check first:

   ```sh
   python scripts/migrate_mongo_to_s3.py
   ```

   Verify the database and document count printed match your deployment.
   The migration requires permission to read the existing collection.
5. Copy the data (this calls Bedrock and writes to S3, which incur usage charges):

   ```sh
   python scripts/migrate_mongo_to_s3.py --apply
   ```

   The script uses stable document IDs, so it can be rerun after an interruption.
   Re-running re-embeds the documents and incurs charges again. It does not delete
   records from either store. If documents change during the copy, rerun it after
   writes stop. This is a copy, not a live sync; keep the old service available.
6. Configure the deployment with `AWS_BEARER_TOKEN_BEDROCK`,
   `S3_VECTOR_BUCKET`, `S3_VECTOR_INDEX`, `BEDROCK_REGION`, and AWS SDK
   credentials authorized for the two read actions above. `BEDROCK_CHAT_MODEL_ID`
   optionally overrides `global.openai.gpt-6-luna`. The deployed API no longer
   needs `MONGODB_URI` or an OpenAI API key. For future content updates, use
   `scripts/index_portfolio.py` with the same S3 and Bedrock settings.
7. Test several representative questions and verify their sources and answers
   before moving traffic to the new deployment. Keep MongoDB until verification
   is complete; switching back to the previous deployment is the rollback.

The Bedrock and S3 API calls, storage, and any data transfer can incur charges.
S3 Vectors is usage based; check the AWS pricing page for the actual Mumbai
rates and your query volume. No live migration or cloud resources are created
by this repository merely by installing it.
