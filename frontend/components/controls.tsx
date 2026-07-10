/**
 * controls.tsx — the small set of form primitives the profile form is built
 * from. Each maps one-to-one to a field in UserProfile and mirrors the
 * ipywidgets prototype (text, number, week slider, diet segmented control,
 * condition chips) but styled to feel modern and calm.
 */
"use client";

import type { DietType, MedicalCondition } from "@/lib/types";
import { DIET_TYPES, MEDICAL_CONDITIONS } from "@/lib/types";

const dietMeta: Record<DietType, { dot: string; short: string }> = {
  "Non-Vegetarian": { dot: "bg-diet-nonveg", short: "Non-Veg" },
  "Ovo-Vegetarian": { dot: "bg-diet-ovo", short: "Ovo-Veg" },
  Vegetarian: { dot: "bg-diet-veg", short: "Veg" },
};

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-sand-500">
      {children}
    </span>
  );
}

const fieldClass =
  "focus-rose w-full rounded-xl border border-sand-200 bg-sand-50 px-3.5 py-2.5 text-sand-800 placeholder:text-sand-400 transition-colors";

export function TextField({
  label,
  value,
  onChange,
  placeholder,
  maxLength,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  maxLength?: number;
}) {
  return (
    <label className="block">
      <Label>{label}</Label>
      <input
        type="text"
        className={fieldClass}
        value={value}
        maxLength={maxLength}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  suffix,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number | "";
  onChange: (v: number | "") => void;
  suffix?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="block">
      <Label>{label}</Label>
      <div className="relative">
        <input
          type="number"
          inputMode="decimal"
          className={`${fieldClass} ${suffix ? "pr-12" : ""}`}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) =>
            onChange(e.target.value === "" ? "" : Number(e.target.value))
          }
        />
        {suffix && (
          <span className="pointer-events-none absolute inset-y-0 right-3.5 flex items-center text-sm text-sand-400">
            {suffix}
          </span>
        )}
      </div>
    </label>
  );
}

export function WeekSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const min = 1;
  const max = 45;
  const pct = ((value - min) / (max - min)) * 100;
  const trimester = value <= 12 ? "1st trimester" : value <= 26 ? "2nd trimester" : "3rd trimester";

  return (
    <div className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <Label>Week of pregnancy</Label>
        <span className="text-sm font-medium text-rose-600">
          Week {value} · <span className="text-sand-500">{trimester}</span>
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="week-range focus-rose h-2 w-full cursor-pointer appearance-none rounded-full"
        style={{
          background: `linear-gradient(to right, var(--rose-500) 0%, var(--rose-500) ${pct}%, var(--sand-200) ${pct}%, var(--sand-200) 100%)`,
        }}
      />
      <style jsx>{`
        .week-range {
          --rose-500: #cf6b8a;
          --sand-200: #e9e3d9;
        }
        .week-range::-webkit-slider-thumb {
          appearance: none;
          width: 22px;
          height: 22px;
          border-radius: 9999px;
          background: #fff;
          border: 3px solid #cf6b8a;
          box-shadow: 0 1px 4px rgba(157, 67, 96, 0.28);
          cursor: pointer;
          transition: transform 0.1s ease;
        }
        .week-range::-webkit-slider-thumb:active {
          transform: scale(1.12);
        }
        .week-range::-moz-range-thumb {
          width: 22px;
          height: 22px;
          border-radius: 9999px;
          background: #fff;
          border: 3px solid #cf6b8a;
          box-shadow: 0 1px 4px rgba(157, 67, 96, 0.28);
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}

export function DietSegmented({
  value,
  onChange,
}: {
  value: DietType;
  onChange: (v: DietType) => void;
}) {
  return (
    <div className="block">
      <Label>Diet</Label>
      <div className="grid grid-cols-3 gap-1.5 rounded-2xl bg-sand-200/60 p-1">
        {DIET_TYPES.map((diet) => {
          const active = diet === value;
          return (
            <button
              key={diet}
              type="button"
              onClick={() => onChange(diet)}
              className={`focus-rose flex items-center justify-center gap-1.5 rounded-xl px-2 py-2 text-sm font-medium transition-all ${
                active
                  ? "bg-sand-50 text-sand-800 shadow-sm"
                  : "text-sand-500 hover:text-sand-700"
              }`}
            >
              <span
                className={`h-2.5 w-2.5 rounded-full ${dietMeta[diet].dot}`}
              />
              {dietMeta[diet].short}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ConditionChips({
  value,
  onChange,
}: {
  value: MedicalCondition[];
  onChange: (v: MedicalCondition[]) => void;
}) {
  const toggle = (c: MedicalCondition) =>
    onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c]);

  return (
    <div className="block">
      <Label>Medical conditions (optional)</Label>
      <div className="flex flex-wrap gap-2">
        {MEDICAL_CONDITIONS.map((c) => {
          const active = value.includes(c);
          return (
            <button
              key={c}
              type="button"
              onClick={() => toggle(c)}
              className={`focus-rose rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors ${
                active
                  ? "border-rose-400 bg-rose-50 text-rose-700"
                  : "border-sand-200 bg-sand-50 text-sand-500 hover:border-sand-300"
              }`}
            >
              {c}
            </button>
          );
        })}
      </div>
    </div>
  );
}
