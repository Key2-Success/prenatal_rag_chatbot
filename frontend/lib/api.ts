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

export async function sendChat(
  message: string,
  profile: UserProfile,
): Promise<ChatResponse> {
  const body: ChatRequest = { message, user_profile: profile };

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
