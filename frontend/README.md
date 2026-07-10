# Poshan Saathi — Frontend

A minimal, single-screen chat interface for the Poshan Saathi prenatal-nutrition
assistant. Built with **Next.js (App Router) + TypeScript + Tailwind**, no
component library — the design is hand-built to stay light and calm.

## What it does

- **Onboarding card → chat.** On first load you fill a short profile (name, age,
  week of pregnancy, diet, weight/height in metric, medical conditions). After
  that the chat is the whole screen; the profile collapses to an editable chip
  in the header.
- **Renders every response type.** The backend returns a `response_type`
  discriminator (`answer` / `emergency` / `out_of_scope` / `no_results`) and the
  UI branches on it — a normal answer shows source citation pills, an emergency
  shows a distinct red safety card, boundary responses show a soft muted note.
  It never parses answer text.
- **Personalized for Indian users**, matching the backend: diet types
  (🔴 Non-Veg · 🟡 Ovo-Veg · 🟢 Veg), the common conditions (low iron,
  hypertension, diabetes), and metric weight/height.

## Run it

The backend must be running first (it's the source of every answer):

```bash
# terminal 1 — backend (from repo root)
uvicorn backend.app.main:app --reload   # serves http://localhost:8000

# terminal 2 — frontend (from frontend/)
npm install
npm run dev                              # serves http://localhost:3000
```

Open http://localhost:3000. The backend's CORS is already pinned to
`http://localhost:3000`, so no extra config is needed for local dev.

To point at a non-default backend, copy `.env.example` to `.env.local` and set
`NEXT_PUBLIC_API_URL`.

## How it maps to the backend

`lib/types.ts` is a hand-kept mirror of `backend/app/models/schemas.py`
(`UserProfile`, `ChatRequest`, `ChatResponse`, `ResponseType`). If a field
changes there, change it here. `lib/api.ts` is the only network boundary — one
`POST /chat` call with typed error handling.

## Scope

Deliberately minimal: no auth, no chat persistence across reloads, no streaming
(the backend returns a single JSON body). The profile lives in memory for the
session only.
