/**
 * ProfileForm — collects the UserProfile. Used in two places: the onboarding
 * card on first load, and the edit sheet in the header. It owns a local draft
 * so edits can be cancelled; the parent only hears about a validated profile
 * on submit.
 *
 * Validation mirrors the Pydantic bounds in schemas.py so we fail early and
 * gently in the UI rather than round-tripping a 422.
 */
"use client";

import { useState } from "react";
import type { DietType, MedicalCondition, UserProfile } from "@/lib/types";
import {
  ConditionChips,
  DietSegmented,
  NumberField,
  TextField,
  WeekSlider,
} from "./controls";

type Draft = {
  name: string;
  age: number | "";
  pregnancy_week: number;
  diet_type: DietType;
  weight_kg: number | "";
  height_cm: number | "";
  medical_conditions: MedicalCondition[];
};

const emptyDraft: Draft = {
  name: "",
  age: "",
  pregnancy_week: 12,
  diet_type: "Vegetarian",
  weight_kg: "",
  height_cm: "",
  medical_conditions: [],
};

function validate(d: Draft): { profile?: UserProfile; error?: string } {
  if (!d.name.trim()) return { error: "Please enter your name." };
  if (d.age === "" || d.age < 10 || d.age > 60)
    return { error: "Please enter an age between 10 and 60." };
  if (d.weight_kg === "" || d.weight_kg <= 0 || d.weight_kg > 300)
    return { error: "Please enter a valid weight in kg." };
  if (d.height_cm === "" || d.height_cm <= 0 || d.height_cm > 250)
    return { error: "Please enter a valid height in cm." };
  return {
    profile: {
      name: d.name.trim(),
      age: d.age,
      pregnancy_week: d.pregnancy_week,
      diet_type: d.diet_type,
      weight_kg: d.weight_kg,
      height_cm: d.height_cm,
      medical_conditions: d.medical_conditions,
    },
  };
}

export function ProfileForm({
  initial,
  submitLabel,
  onSubmit,
  onCancel,
}: {
  initial?: UserProfile;
  submitLabel: string;
  onSubmit: (p: UserProfile) => void;
  onCancel?: () => void;
}) {
  const [draft, setDraft] = useState<Draft>(
    initial ? { ...initial } : emptyDraft,
  );
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof Draft>(key: K, value: Draft[K]) =>
    setDraft((d) => ({ ...d, [key]: value }));

  const handleSubmit = () => {
    const { profile, error } = validate(draft);
    if (error) {
      setError(error);
      return;
    }
    setError(null);
    onSubmit(profile!);
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <TextField
          label="Name"
          value={draft.name}
          onChange={(v) => set("name", v)}
          placeholder="Your name"
          maxLength={100}
        />
        <NumberField
          label="Age"
          value={draft.age}
          onChange={(v) => set("age", v)}
          min={10}
          max={60}
        />
      </div>

      <WeekSlider
        value={draft.pregnancy_week}
        onChange={(v) => set("pregnancy_week", v)}
      />

      <DietSegmented
        value={draft.diet_type}
        onChange={(v) => set("diet_type", v)}
      />

      <div className="grid grid-cols-2 gap-3">
        <NumberField
          label="Weight"
          value={draft.weight_kg}
          onChange={(v) => set("weight_kg", v)}
          suffix="kg"
          min={1}
          max={300}
          step={0.1}
        />
        <NumberField
          label="Height"
          value={draft.height_cm}
          onChange={(v) => set("height_cm", v)}
          suffix="cm"
          min={1}
          max={250}
          step={0.1}
        />
      </div>

      <ConditionChips
        value={draft.medical_conditions}
        onChange={(v) => set("medical_conditions", v)}
      />

      {error && (
        <p className="text-sm font-medium text-rose-600" role="alert">
          {error}
        </p>
      )}

      <div className="flex gap-3">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="focus-rose flex-1 rounded-xl border border-sand-200 bg-sand-50 px-4 py-3 font-medium text-sand-600 transition-colors hover:bg-sand-100"
          >
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={handleSubmit}
          className="focus-rose flex-1 rounded-xl bg-rose-600 px-4 py-3 font-semibold text-sand-50 shadow-sm transition-colors hover:bg-rose-700"
        >
          {submitLabel}
        </button>
      </div>
    </div>
  );
}
