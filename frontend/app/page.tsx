/**
 * page.tsx — top-level orchestration.
 *
 * Two states, one transition:
 *   1. No profile yet  → centered onboarding card (the six-field form).
 *   2. Profile set     → chat is the whole screen; profile collapses to an
 *      editable chip in the header (name · week · diet).
 *
 * All state is client-side and ephemeral by design — no persistence, no auth
 * (see frontend/README.md "Scope"). The profile lives in memory for the
 * session and is attached to every /chat request.
 */
"use client";

import { useState } from "react";
import { Chat } from "@/components/Chat";
import { ProfileForm } from "@/components/ProfileForm";
import { ProfileSheet } from "@/components/ProfileSheet";
import { Wordmark } from "@/components/Wordmark";
import type { UserProfile } from "@/lib/types";

const dietDot: Record<UserProfile["diet_type"], string> = {
  "Non-Vegetarian": "bg-diet-nonveg",
  "Ovo-Vegetarian": "bg-diet-ovo",
  Vegetarian: "bg-diet-veg",
};

export default function Home() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [editing, setEditing] = useState(false);

  // --- Onboarding ---
  if (!profile) {
    return (
      <main className="mx-auto flex min-h-screen max-w-xl flex-col justify-center px-5 py-10">
        <div className="mb-6 flex flex-col items-center text-center">
          <Wordmark />
          <p className="mt-5 max-w-sm text-sand-600">
            A gentle, evidence-based nutrition companion for your pregnancy —
            grounded in guidance from India's health authorities. Tell me a
            little about you to begin.
          </p>
        </div>
        <div className="rounded-3xl border border-sand-200 bg-sand-50/80 p-6 shadow-sm sm:p-7">
          <ProfileForm
            submitLabel="Start chatting"
            onSubmit={(p) => setProfile(p)}
          />
        </div>
      </main>
    );
  }

  // --- Chat ---
  return (
    <main className="mx-auto flex h-screen max-w-2xl flex-col px-4">
      <header className="flex items-center justify-between py-3">
        <Wordmark compact />
        <button
          onClick={() => setEditing(true)}
          className="focus-rose flex items-center gap-2 rounded-full border border-sand-200 bg-sand-50 py-1.5 pl-3 pr-2.5 text-sm text-sand-600 shadow-sm transition-colors hover:bg-sand-100"
        >
          <span className="font-medium text-sand-700">
            {profile.name.split(" ")[0]}
          </span>
          <span className="text-sand-400">· wk {profile.pregnancy_week} ·</span>
          <span className={`h-2 w-2 rounded-full ${dietDot[profile.diet_type]}`} />
          <span className="text-sand-400">✎</span>
        </button>
      </header>

      <div className="min-h-0 flex-1 pb-3">
        <Chat profile={profile} />
      </div>

      {editing && (
        <ProfileSheet
          profile={profile}
          onSave={(p) => {
            setProfile(p);
            setEditing(false);
          }}
          onClose={() => setEditing(false)}
        />
      )}
    </main>
  );
}
