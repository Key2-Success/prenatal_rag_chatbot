/**
 * Chat — the conversation thread and composer.
 *
 * Rendering branches on ChatResponse.response_type (never on answer text),
 * matching the contract documented in schemas.py:
 *   answer       → assistant bubble + source citation pills
 *   emergency    → distinct red safety card (call 108)
 *   out_of_scope → soft muted boundary note
 *   no_results   → soft muted boundary note
 */
"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChatError, sendChat } from "@/lib/api";
import type { ChatResponse, Source, UserProfile } from "@/lib/types";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; response: ChatResponse }
  | { role: "error"; text: string };

/**
 * SourcePill — a compact `FOGSI · p.8` bubble. On hover (desktop) or tap
 * (mobile) it reveals a card with the document title, the cross-encoder
 * relevance score, and the full retrieved passage — "showing the RAG working".
 *
 * The card is rendered through a portal to document.body and positioned with
 * `fixed` from the pill's bounding rect, so the thread's scroll container
 * (overflow-y-auto) can't clip it. A short hover-intent delay lets the pointer
 * travel from the pill into the card to scroll a long passage.
 */
function SourcePill({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const pillRef = useRef<HTMLButtonElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pct = Math.round(source.relevance_score * 100);

  const show = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    const r = pillRef.current?.getBoundingClientRect();
    if (r) setPos({ left: r.left + r.width / 2, top: r.top });
    setOpen(true);
  };
  const scheduleHide = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = setTimeout(() => setOpen(false), 120);
  };

  // A fixed-position card goes stale if the page scrolls; close it instead.
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  return (
    <>
      <button
        ref={pillRef}
        type="button"
        onMouseEnter={show}
        onMouseLeave={scheduleHide}
        onFocus={show}
        onBlur={scheduleHide}
        onClick={() => (open ? setOpen(false) : show())}
        aria-label={`Source: ${source.org_display_name}, page ${source.page}. ${pct}% relevance. Show passage.`}
        className="focus-rose inline-flex shrink-0 items-center gap-1 rounded-full border border-sand-200 bg-sand-50 px-2.5 py-1 text-xs text-sand-600 transition-colors hover:border-rose-300 hover:bg-rose-50"
      >
        <span className="font-semibold text-rose-600">
          {source.org_display_name}
        </span>
        <span className="text-sand-400">·</span>
        <span>p.{source.page}</span>
      </button>

      {open &&
        pos &&
        createPortal(
          <div
            style={{
              position: "fixed",
              left: pos.left,
              top: pos.top - 10,
              transform: "translate(-50%, -100%)",
            }}
            onMouseEnter={show}
            onMouseLeave={scheduleHide}
            className="z-50 w-72 max-w-[calc(100vw-2rem)] animate-fade-up rounded-xl border border-sand-200 bg-sand-50 p-3 shadow-xl"
          >
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-rose-600">
                {source.org_display_name} · p.{source.page}
              </span>
              <span className="shrink-0 rounded-full bg-rose-100 px-1.5 py-0.5 text-[0.65rem] font-semibold text-rose-700">
                {pct}% match
              </span>
            </div>
            <div className="mb-2 text-xs font-medium text-sand-700">
              {source.doc_title}{" "}
              <span className="text-sand-400">· {source.year_published}</span>
            </div>
            <p className="max-h-44 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-sand-600">
              {source.chunk_text}
            </p>
            {/* downward arrow toward the pill */}
            <div className="absolute left-1/2 top-full h-2.5 w-2.5 -translate-x-1/2 -translate-y-[6px] rotate-45 border-b border-r border-sand-200 bg-sand-50" />
          </div>,
          document.body,
        )}
    </>
  );
}

function AssistantMessage({ response }: { response: ChatResponse }) {
  if (response.response_type === "emergency") {
    return (
      <div className="max-w-[85%] animate-fade-up rounded-2xl rounded-tl-md border border-red-200 bg-red-50 px-4 py-3 text-red-800 shadow-sm">
        <div className="mb-1 flex items-center gap-1.5 text-sm font-semibold">
          <span aria-hidden>⚠</span> Please seek care now
        </div>
        <p className="text-[0.95rem] leading-relaxed">{response.answer}</p>
      </div>
    );
  }

  if (
    response.response_type === "out_of_scope" ||
    response.response_type === "no_results"
  ) {
    return (
      <div className="max-w-[85%] animate-fade-up rounded-2xl rounded-tl-md border border-sand-200 bg-sand-100/70 px-4 py-3 text-sand-600 shadow-sm">
        <p className="text-[0.95rem] leading-relaxed">{response.answer}</p>
      </div>
    );
  }

  // answer
  return (
    <div className="max-w-[85%] animate-fade-up space-y-2">
      <div className="rounded-2xl rounded-tl-md bg-sand-50 px-4 py-3 text-sand-800 shadow-sm ring-1 ring-sand-200/70">
        <p className="whitespace-pre-wrap text-[0.95rem] leading-relaxed">
          {response.answer}
        </p>
      </div>
      {response.sources.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pl-1">
          {/* Display sources by descending match % (relevance). Copy before
              sorting so we don't mutate the response; the backend keeps them in
              source-priority order (which the LLM context + routing eval rely
              on) — this only changes the visual order of the pills. Stable sort
              keeps priority order as the tiebreak for equal scores. */}
          {[...response.sources]
            .sort((a, b) => b.relevance_score - a.relevance_score)
            .map((s, i) => (
              <SourcePill key={`${s.org_display_name}-${s.page}-${i}`} source={s} />
            ))}
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <div className="max-w-[85%] animate-fade-up rounded-2xl rounded-tl-md bg-sand-50 px-4 py-3.5 shadow-sm ring-1 ring-sand-200/70">
      <div className="flex gap-1.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-2 w-2 animate-blink rounded-full bg-rose-400"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </div>
    </div>
  );
}

export function Chat({ profile }: { profile: UserProfile }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);
    try {
      const response = await sendChat(text, profile);
      setMessages((m) => [...m, { role: "assistant", response }]);
    } catch (e) {
      const msg =
        e instanceof ChatError
          ? e.message
          : "Something went wrong. Please try again.";
      setMessages((m) => [...m, { role: "error", text: msg }]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Thread */}
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-1 py-4">
        {messages.length === 0 && (
          <div className="mx-auto mt-10 max-w-sm text-center">
            <p className="text-sand-500">
              Hi {profile.name.split(" ")[0]} — ask me anything about your
              nutrition during pregnancy. Try{" "}
              <button
                className="font-medium text-rose-600 underline decoration-rose-200 underline-offset-2 hover:decoration-rose-400"
                onClick={() => setInput("How much iron should I be eating?")}
              >
                "How much iron should I be eating?"
              </button>
            </p>
          </div>
        )}

        {messages.map((m, i) => {
          if (m.role === "user") {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] animate-fade-up rounded-2xl rounded-tr-md bg-rose-600 px-4 py-3 text-[0.95rem] leading-relaxed text-sand-50 shadow-sm">
                  {m.text}
                </div>
              </div>
            );
          }
          if (m.role === "error") {
            return (
              <div key={i} className="flex justify-start">
                <div className="max-w-[85%] animate-fade-up rounded-2xl rounded-tl-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 shadow-sm">
                  {m.text}
                </div>
              </div>
            );
          }
          return (
            <div key={i} className="flex justify-start">
              <AssistantMessage response={m.response} />
            </div>
          );
        })}

        {loading && (
          <div className="flex justify-start">
            <TypingDots />
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-sand-200/80 bg-sand-100/60 px-1 pt-3">
        <div className="flex items-end gap-2 rounded-2xl border border-sand-200 bg-sand-50 p-2 shadow-sm focus-within:border-rose-300">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            maxLength={1000}
            placeholder="Ask about your nutrition…"
            className="max-h-32 flex-1 resize-none bg-transparent px-2.5 py-2 text-sand-800 placeholder:text-sand-400 focus:outline-none"
          />
          <button
            onClick={send}
            disabled={!input.trim() || loading}
            className="focus-rose grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-rose-600 text-sand-50 transition-colors hover:bg-rose-700 disabled:cursor-not-allowed disabled:bg-sand-300"
            aria-label="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M4 12L20 4L14 20L11 13L4 12Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
        <p className="px-2 py-1.5 text-center text-xs text-sand-400">
          Poshan Saathi offers general guidance, not medical advice. For
          emergencies, call 108.
        </p>
      </div>
    </div>
  );
}
