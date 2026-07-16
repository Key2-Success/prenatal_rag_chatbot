/**
 * api.ts — the single network boundary to the FastAPI backend.
 *
 * One function, POST /chat. The backend validates the body against the
 * ChatRequest Pydantic model (422 on bad input) and returns a ChatResponse.
 * We surface a typed error the UI can render calmly rather than throwing
 * raw fetch failures at the user.
 */
import type { ChatRequest, ChatResponse, UserProfile } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ChatError extends Error {}

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
): Promise<ChatResponse> {
  const body: ChatRequest = { message, user_profile: profile };

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
