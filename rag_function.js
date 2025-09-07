// Atlas HTTPS Function: ragQuery
// Expects POST JSON: { q: string, k?: number }
// Returns: { answer: string, sources: [ { id, text, meta, score } ] }

exports = async function(payload, response) {
  try {
    const body = payload.body ? JSON.parse(payload.body.text()) : {};
    const q = (body.q || "").trim();
    if (!q) {
      response.setStatusCode(400);
      return { error: "q required" };
    }
    const k = body.k ? parseInt(body.k, 10) : 4;

    const OPENAI_KEY = context.values.get("OPENAI_API_KEY");
    if (!OPENAI_KEY) {
      response.setStatusCode(500);
      return { error: "OPENAI_API_KEY not configured in App Services values" };
    }

    // 1) embed the query
    const embResp = await context.http.post({
      url: "https://api.openai.com/v1/embeddings",
      headers: {
        "Content-Type": ["application/json"],
        "Authorization": [`Bearer ${OPENAI_KEY}`]
      },
      body: JSON.stringify({ model: "text-embedding-3-small", input: q }),
      encodeBodyAsJSON: false
    });

    const embText = embResp.body.text();
    const embJson = JSON.parse(embText);
    const qEmb = embJson?.data?.[0]?.embedding;
    if (!qEmb) throw new Error("embedding failed");

    // 2) run Atlas Search KNN
    // Update `your_db_name` and `chunks` if different; make sure a Search index exists mapping `embedding` as vector
    const coll = context.services.get("mongodb-atlas").db(context.values.get("MONGODB_DB") || "resume_rag").collection(context.values.get("MONGODB_COLL") || "chunks");

    // Try a $search aggregation (knnBeta / $vectorSearch depending on atlas version)
    let results = [];
    try {
      const pipeline = [
        {
          $search: {
            knnBeta: {
              vector: qEmb,
              path: "embedding",
              k: k
            }
          }
        },
        {
          $project: {
            _id: 0,
            id: "$id",
            text: 1,
            meta: 1,
            score: { $meta: "searchScore" }
          }
        },
        { $limit: k }
      ];
      results = await coll.aggregate(pipeline).toArray();
    } catch (err) {
      // fallback: compute cosine similarity in JS across collection (safe for small datasets)
      const docs = await coll.find({}, { projection: { id: 1, text: 1, meta: 1, embedding: 1 } }).toArray();
      function cosine(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
        if (na === 0 || nb === 0) return 0;
        return dot / (Math.sqrt(na)*Math.sqrt(nb));
      }
      const scored = docs.map(d => ({ id: d.id, text: d.text, meta: d.meta, score: d.embedding ? cosine(qEmb, d.embedding) : 0 }));
      scored.sort((a,b) => b.score - a.score);
      results = scored.slice(0, k);
    }

    // 3) assemble context and call Chat Completion
    const contextParts = results.map((r, i) => `SOURCE ${i+1} (score:${(r.score||0).toFixed(4)}):\n${r.text}`).join("\n\n---\n\n");
    const system = "You are a helpful assistant answering questions about a person's resume/portfolio. Use the CONTEXT to answer precisely; if not present, be honest and say you don't know.";
    const messages = [
      { role: "system", content: system },
      { role: "system", content: `CONTEXT:\n${contextParts}` },
      { role: "user", content: q }
    ];

    // call Chat Completion
    const chatResp = await context.http.post({
      url: "https://api.openai.com/v1/chat/completions",
      headers: {
        "Content-Type": ["application/json"],
        "Authorization": [`Bearer ${OPENAI_KEY}`]
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages,
        max_tokens: 600,
        temperature: 0.0
      }),
      encodeBodyAsJSON: false
    });

    const chatText = chatResp.body.text();
    const chatJson = JSON.parse(chatText);
    const answer = chatJson?.choices?.[0]?.message?.content ?? "No answer";

    return { answer, sources: results };
  } catch (err) {
    console.error("ragQuery error:", err);
    response.setStatusCode(500);
    return { error: String(err) };
  }
};
