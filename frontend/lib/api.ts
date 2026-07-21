/**
 * api.ts — the single network boundary to the FastAPI backend.
 *
 * One function, POST /chat. The backend validates the body against the
 * ChatRequest Pydantic model (422 on bad input) and returns a ChatResponse.
 * We surface a typed error the UI can render calmly rather than throwing
 * raw fetch failures at the user.
 */
import type { ChatRequest, ChatResponse, ChatTurn, UserProfile } from "./types";

// History caps, kept in lockstep with schemas.py (ChatTurn / ChatRequest):
// send at most the last 6 turns (3 exchanges — backend accepts 8; we stay
// under), each clipped to the backend's 1500-char per-turn limit so a long
// assistant answer can never 422 the request.
const HISTORY_MAX_TURNS = 6;
const HISTORY_MAX_CHARS = 1500;

/** Clip the transcript tail to the backend's history caps. */
function trimHistory(history: ChatTurn[]): ChatTurn[] {
  return history.slice(-HISTORY_MAX_TURNS).map((t) => ({
    role: t.role,
    content:
      t.content.length > HISTORY_MAX_CHARS
        ? t.content.slice(0, HISTORY_MAX_CHARS)
        : t.content,
  }));
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ChatError extends Error {}

/**
 * Fire-and-forget wake-up ping, called when the landing page loads. On Render's
 * free tier the backend sleeps after ~15 min idle and takes ~1 min to wake, so
 * we kick off that wake-up *while the user fills out the profile form* — by the
 * time they hit "Start chatting" it's warm. Hits /health only (no OpenAI /
 * Pinecone work, so it's effectively free) and swallows all errors — it's purely
 * opportunistic, never blocks or surfaces anything to the user.
 */
export function warmUpBackend(): void {
  void fetch(`${API_BASE}/health`, { method: "GET" }).catch(() => {});
}

/**
 * A stable, anonymous per-browser id kept in localStorage. Sent as X-Anon-Id so
 * Langfuse can count unique visitors and group their sessions — no login, no PII,
 * just a random UUID this browser reuses. Falls back gracefully if storage is
 * blocked (private mode) or we're not in a browser.
 */
function getAnonId(): string {
  if (typeof window === "undefined") return "server";
  try {
    let id = localStorage.getItem("psaathi_anon_id");
    if (!id) {
      id = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2);
      localStorage.setItem("psaathi_anon_id", id);
    }
    return id;
  } catch {
    return "no-storage";
  }
}

export async function sendChat(
  message: string,
  profile: UserProfile,
  history: ChatTurn[] = [],
): Promise<ChatResponse> {
  const body: ChatRequest = {
    message,
    user_profile: profile,
    history: trimHistory(history),
  };

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Anon-Id": getAnonId(),
      },
      body: JSON.stringify(body),
    });
  } catch {
    // Network-level failure — backend down, CORS, offline.
    throw new ChatError(
      "I couldn't reach the assistant. Please check that the backend is running.",
    );
  }

  if (!res.ok) {
    // 422 (validation) or 500 (pipeline error). The backend never leaks
    // internals on 500; 422 carries field detail we don't surface verbatim.
    throw new ChatError(
      res.status === 422
        ? "Something about your profile or message wasn't valid. Please review and try again."
        : "The assistant ran into a problem. Please try again in a moment.",
    );
  }

  return (await res.json()) as ChatResponse;
}
