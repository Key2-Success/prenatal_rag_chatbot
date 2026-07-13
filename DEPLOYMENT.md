# Deployment

Poshan Saathi deploys as two pieces:

| Piece | Host | Why |
|---|---|---|
| **Frontend** (Next.js) | **Vercel** | Purpose-built for Next.js; free tier. |
| **Backend** (FastAPI) | **Render** (Docker) | A persistent Python server. Runs on the **free** tier because it uses Pinecone's **hosted** reranker (`RERANKER_BACKEND=pinecone`) — no torch/model, so the image stays small. |

**Cost:** effectively just OpenAI usage (~$0.002/answer), and that's capped by the
built-in rate limits + daily budget. Render free, Vercel free, Pinecone free tier
(hosted rerank: 500/month), Langfuse free. **Set an OpenAI hard usage limit as the
ultimate backstop (step 4).**

---

## 1. Backend → Render

The repo already contains everything Render needs: `Dockerfile`, `.dockerignore`,
`requirements-prod.txt` (runtime-only deps), and `render.yaml` (blueprint).

1. Push this repo to GitHub (already done).
2. Render Dashboard → **New → Blueprint** → connect the repo. Render reads
   `render.yaml`, builds the `Dockerfile`, and creates a **free** web service with
   a `/health` check.
3. When prompted, set the env vars marked `sync: false`:
   - `OPENAI_API_KEY` — your key
   - `PINECONE_API_KEY` — your key
   - `CORS_ALLOW_ORIGINS` — leave blank for now; fill in after step 2 (your Vercel URL)
   - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional (omit to disable tracing)
   - `RERANKER_BACKEND=pinecone` and `LANGFUSE_HOST` are already set in the blueprint.
4. Deploy. Note the service URL, e.g. `https://poshan-saathi-api.onrender.com`.
   Verify: `curl https://<your-service>.onrender.com/health` → `{"status":"ok"}`.

> **Free-tier note:** the service spins down after ~15 min idle; the next request
> pays a ~30–60s cold start. Fine for a demo — upgrade the plan for always-on.

Prefer Railway or Fly instead? Same Dockerfile works — set the same env vars and
point the service at it. (Fly: `fly launch` reads the Dockerfile; set `PORT` is
handled automatically.)

## 2. Frontend → Vercel

1. Vercel → **New Project** → import the repo, set **Root Directory = `frontend`**.
2. Add an env var: `NEXT_PUBLIC_API_URL = https://<your-render-service>.onrender.com`
   (the URL from step 1).
3. Deploy. Note the Vercel URL, e.g. `https://poshan-saathi.vercel.app`.

## 3. Wire CORS back to the frontend

On Render, set **`CORS_ALLOW_ORIGINS`** to your Vercel URL
(e.g. `https://poshan-saathi.vercel.app`) and redeploy the backend. This is what
lets the browser call the API. Comma-separate if you have multiple origins
(e.g. a preview domain).

## 4. Set an OpenAI hard usage limit (do this!)

In the OpenAI dashboard → **Billing → Limits**, set a monthly **hard** cap. The
app's rate limits + daily budget cap cost at the edge, but this is enforced by
OpenAI itself and catches anything the app can't see (a bug, a second service, a
retry storm). It is the real guarantee that this can't run up your bill.

---

## What protects the public endpoint (already built in)

- **Per-IP rate limit:** 10/min + 50/day (env: `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY`).
- **Global daily budget:** 500/day across all clients → 503 when hit (`DAILY_REQUEST_BUDGET`).
- **Body-size cap** 16KB (`MAX_BODY_BYTES`) + message `max_length=500`.
- **CORS** locked to an explicit allow-list (`CORS_ALLOW_ORIGINS`).
- Real client IP read from `X-Forwarded-For` (correct behind Render's proxy).

## Local development is unchanged

`RERANKER_BACKEND` defaults to `local` (self-hosted model on your Apple Silicon),
and CORS defaults to `http://localhost:3000`. Run the backend with
`uvicorn backend.app.main:app --reload` and the frontend with `npm run dev` —
no deployment env vars needed.

## Ingestion is not part of the deployment

The corpus is already upserted to Pinecone. The deployed image intentionally
excludes the ingest pipeline (LlamaParse / langchain / the source PDFs). To
re-ingest, run `python -m scripts.ingest` locally with the full `requirements.txt`.
