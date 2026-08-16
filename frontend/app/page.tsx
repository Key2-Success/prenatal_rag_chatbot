/**
 * page.tsx — project landing page.
 *
 * The public entry point: explains what Poshan Saathi is, the problem it
 * solves, and how it was built, then routes visitors into the app at /chat.
 * Written for recruiters and hiring managers arriving cold from a link.
 *
 * Laid out as a 2x2 card grid so the three CTAs stay visible without
 * scrolling on a typical laptop viewport.
 *
 * It also pings the backend on mount — Render's free tier sleeps after ~15 min
 * idle, so waking it while someone reads this page means the app is warm by the
 * time they click through.
 */
"use client";

import { useEffect } from "react";
import { Wordmark } from "@/components/Wordmark";
import { warmUpBackend } from "@/lib/api";

const GITHUB_README =
  "https://github.com/Key2-Success/prenatal_rag_chatbot/blob/main/README.md";
const DEMO_VIDEO =
  "https://drive.google.com/file/d/1TLB4YFKsw1b1tXJGa2Z5Yz9gbQVlRjzJ/view";

// Flattened from the README stack table — frontend, backend, AI/RAG,
// observability/eval, then dev tooling.
// Split across two explicit lines so the break lands where we want it
// (after Upstash Redis) rather than wherever the text happens to wrap.
const TECH_LINE_1 = [
  "Python", "FastAPI", "LangChain", "LlamaIndex", "Langfuse",
  "Pinecone", "BM25", "RAGAS", "OpenAI", "Claude", "Upstash Redis",
].join(", ");
const TECH_LINE_2 = [
  "Docker", "Render",
  "Next.js", "React", "TypeScript", "Tailwind CSS", "Vercel",
  "Claude Code", "Git", "GitHub",
].join(", ");

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-sand-200 bg-sand-50/80 p-4 shadow-sm">
      <h2 className="font-serif text-base font-semibold text-sand-800">
        {title}
      </h2>
      <p className="mt-1.5 text-sm leading-relaxed text-sand-700">{children}</p>
    </section>
  );
}

export default function Landing() {
  // Warm the backend while the visitor reads — see file header.
  useEffect(() => {
    warmUpBackend();
  }, []);

  return (
    <main className="mx-auto max-w-6xl px-6 py-5 sm:py-6">
      <div className="flex flex-col items-center text-center">
        <Wordmark />
        <p className="mt-3 text-sand-600">Welcome to my AI RAG project! 👋</p>
      </div>

      {/* Flex row gives a hanging indent: wrapped lines align under the start
          of the list rather than under the "Technology Stack:" label. */}
      <p className="mt-3 flex gap-2 text-base leading-relaxed text-sand-600">
        <span className="whitespace-nowrap font-semibold text-sand-700">
          Technology Stack:
        </span>
        <span>
          {TECH_LINE_1},
          <br />
          {TECH_LINE_2}
        </span>
      </p>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row">
        <a
          href="/chat"
          className="focus-rose rounded-2xl bg-rose-600 px-6 py-3 text-center font-semibold text-white shadow-sm transition-colors hover:bg-rose-700 sm:flex-[2]"
        >
          Try it yourself here →
        </a>
        <a
          href={GITHUB_README}
          target="_blank"
          rel="noopener noreferrer"
          className="focus-rose rounded-2xl border border-sand-200 bg-sand-50 px-6 py-3 text-center font-medium text-sand-700 shadow-sm transition-colors hover:bg-sand-100 sm:flex-1"
        >
          How I built it
        </a>
        <a
          href={DEMO_VIDEO}
          target="_blank"
          rel="noopener noreferrer"
          className="focus-rose rounded-2xl border border-sand-200 bg-sand-50 px-6 py-3 text-center font-medium text-sand-700 shadow-sm transition-colors hover:bg-sand-100 sm:flex-1"
        >
          Chatbot demo
        </a>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Card title="TL;DR">
          <strong className="font-semibold text-sand-800">Poshan Saathi</strong>{" "}
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
          I built a system designed specifically for the Indian pregnant woman,
          sourcing nutritional guidance directly from India's Ministry of
          Health, India's FOGSI obstetrics federation, and the WHO as a
          fallback, while tailoring every answer to the woman's diet, trimester,
          and medical conditions.
        </Card>

        <Card title="Why this problem">
          I was a quarterfinalist for The Gates Foundation's AI Fellow Program
          (top 20/4500+ applicants, ie top 0.4%) where we were asked to build a
          prototype of an antenatal chatbot. I chose to bring my prototype to
          production using state-of-the-art RAG techniques to upskill my AI
          skillset and to bring a working solution to a real problem! Here's a
          peek into how I approached this project. 🥰
        </Card>
      </div>

      <p className="mt-3 text-center text-xs leading-relaxed text-sand-500">
        Educational portfolio project — not medical advice. Always consult your
        doctor or midwife.
      </p>
    </main>
  );
}
