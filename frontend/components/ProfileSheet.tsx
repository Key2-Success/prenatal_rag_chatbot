/**
 * ProfileSheet — a slide-over that lets the user edit their profile after
 * onboarding, opened from the header chip. Kept out of the way so the chat
 * stays the focus; the profile is set once and rarely touched.
 */
"use client";

import { useEffect } from "react";
import type { UserProfile } from "@/lib/types";
import { ProfileForm } from "./ProfileForm";

export function ProfileSheet({
  profile,
  onSave,
  onClose,
}: {
  profile: UserProfile;
  onSave: (p: UserProfile) => void;
  onClose: () => void;
}) {
  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Scrim */}
      <div
        className="absolute inset-0 bg-sand-900/20 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* Panel */}
      <div className="relative flex h-full w-full max-w-md animate-fade-up flex-col overflow-y-auto bg-sand-100 shadow-2xl">
        <div className="flex items-center justify-between border-b border-sand-200 px-6 py-4">
          <h2 className="font-serif text-xl font-semibold tracking-tight text-sand-800">
            Your profile
          </h2>
          <button
            onClick={onClose}
            className="focus-rose grid h-8 w-8 place-items-center rounded-lg text-sand-500 hover:bg-sand-200"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-6">
          <ProfileForm
            initial={profile}
            submitLabel="Save changes"
            onSubmit={onSave}
            onCancel={onClose}
          />
        </div>
      </div>
    </div>
  );
}
