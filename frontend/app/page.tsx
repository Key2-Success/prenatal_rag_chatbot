/**
 * page.tsx — project landing page.
 *
 * The public entry point: explains what Poshan Saathi is, the problem it
 * solves, and how it was built, then routes visitors into the app at /chat.
 * Written for recruiters and hiring managers arriving cold from a link.
 *
 * Type: all-sans (Nunito) with small uppercase tracked labels for the card
 * headings — the serif is reserved for the wordmark so the page reads as
 * editorial rather than clunky. Laid out as a 2x2 card grid so the three CTAs
 * stay visible without scrolling on a typical laptop viewport.
 *
 * It also pings the backend on mount — Render's free tier sleeps after ~15 min
 * idle, so waking it while someone reads this page means the app is warm by the
 * time they click through.
 */
"use client";

import { useEffect } from "react";
import { MotherAppleMark } from "@/components/Wordmark";
import { warmUpBackend } from "@/lib/api";

const GITHUB_README =
  "https://github.com/Key2-Success/prenatal_rag_chatbot/blob/main/README.md";
const DEMO_VIDEO =
  "https://drive.google.com/file/d/1TLB4YFKsw1b1tXJGa2Z5Yz9gbQVlRjzJ/view";

// Rendered as pills rather than a comma-separated sentence: 21 items reads as
// a wall of prose, but as chips it scans like metadata and stays out of the
// way of the copy.
const TECHNOLOGIES = [
  "Python", "FastAPI", "LangChain", "LlamaIndex", "Langfuse",
  "Pinecone", "BM25", "RAGAS", "OpenAI", "Claude", "Upstash Redis",
  "Docker", "Render",
  "Next.js", "React", "TypeScript", "Tailwind CSS", "Vercel",
  "Claude Code", "Git",
];

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-sand-200/70 bg-white/60 p-4 shadow-[0_1px_2px_rgba(36,31,27,0.04)] transition-shadow hover:shadow-[0_2px_10px_rgba(36,31,27,0.06)]">
      <h2 className="text-[0.7rem] font-bold uppercase tracking-[0.14em] text-rose-600">
        {title}
      </h2>
      <p className="mt-2 text-[0.9rem] leading-[1.65] text-sand-700">
        {children}
      </p>
    </section>
  );
}

export default function Landing() {
  // Warm the backend while the visitor reads — see file header.
  useEffect(() => {
    warmUpBackend();
  }, []);

  return (
    <main className="mx-auto max-w-4xl px-6 py-4 sm:py-5">
      {/* Custom lockup (rather than <Wordmark/>) so the mark scales to the full
          height of BOTH text lines and the welcome line aligns with the name. */}
      <div className="flex justify-center">
        <div className="flex items-center gap-3">
          <div
            className="grid h-[3.4rem] w-[3.4rem] shrink-0 place-items-center rounded-2xl bg-rose-600 text-sand-50 shadow-sm"
            aria-hidden
          >
            <MotherAppleMark className="h-11 w-11" />
          </div>
          <div className="leading-tight">
            <div className="flex items-baseline gap-2">
              <span className="font-sans text-xl font-bold tracking-tight text-sand-800">
                Poshan Saathi
              </span>
              <span className="text-sm text-sand-600">
                <span className="text-sand-300">|</span> nutrition companion
              </span>
            </div>
            <p className="mt-1 text-[0.95rem] tracking-wide text-sand-600">
              Welcome to my AI RAG project 👋
            </p>
          </div>
        </div>
      </div>

      {/* Stack as pills — scans as metadata instead of a paragraph of commas. */}
      <div className="mx-auto mt-4 flex max-w-4xl flex-wrap items-center justify-center gap-1.5">
        <span className="mr-1 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-sand-500">
          Technology Stack
        </span>
        {TECHNOLOGIES.map((tech) => (
          <span
            key={tech}
            className="rounded-full border border-sand-200/80 bg-white/70 px-2.5 py-[3px] text-[0.78rem] font-medium text-sand-700"
          >
            {tech}
          </span>
        ))}
      </div>

      <div className="mt-4 flex flex-col gap-2.5 sm:flex-row">
        <a
          href="/chat"
          className="focus-rose rounded-xl bg-rose-600 px-6 py-3 text-center text-[0.9rem] font-semibold tracking-wide text-white shadow-sm transition-all hover:bg-rose-700 hover:shadow-md sm:flex-1"
        >
          Try it yourself here →
        </a>
        <a
          href={GITHUB_README}
          target="_blank"
          rel="noopener noreferrer"
          className="focus-rose rounded-xl border border-rose-300 bg-white px-6 py-3 text-center text-[0.9rem] font-semibold tracking-wide text-rose-700 transition-colors hover:border-rose-400 hover:bg-rose-50 sm:flex-1"
        >
          How I built it ↗
        </a>
        <a
          href={DEMO_VIDEO}
          target="_blank"
          rel="noopener noreferrer"
          className="focus-rose rounded-xl border border-rose-300 bg-white px-6 py-3 text-center text-[0.9rem] font-semibold tracking-wide text-rose-700 transition-colors hover:border-rose-400 hover:bg-rose-50 sm:flex-1"
        >
          Chatbot demo ↗
        </a>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Card title="TL;DR">
          <strong className="font-semibold text-sand-700">Poshan Saathi</strong>{" "}
          ("nutrition companion" in Hindi) is a RAG chatbot I built to answer
          pregnancy-nutrition questions for women in India.
        </Card>

        <Card title="The problem">
          The Indian diaspora has specific dietary requirements (ovo-vegetarian
          and vegetarian are common) and medical conditions (anemia and diabetes
          are common), while most maternal nutrition advice is generic,
          unsourced, or written for a Western diet.
        </Card>

        <Card title="My solution">
          I built a system specifically for the Indian pregnant woman, sourcing
          nutritional guidance directly from India's Ministry of Health, India's
          FOGSI obstetrics federation, and the WHO as a fallback, while
          tailoring every answer to the woman's diet, trimester, and medical
          conditions.
        </Card>

        <Card title="Why this problem">
          I was a quarterfinalist for The Gates Foundation's AI Fellow Program
          (top 20/4500+ applicants, ie top 0.4%) where we built a prototype of
          an antenatal chatbot. I chose to bring my prototype to production
          using state-of-the-art RAG techniques to upskill my AI skillset to a
          real problem!
        </Card>
      </div>

      <p className="mt-3 text-center text-[0.72rem] tracking-wide text-sand-500">
        Educational portfolio project — not medical advice. Always consult your
        doctor or midwife.
      </p>
    </main>
  );
}
