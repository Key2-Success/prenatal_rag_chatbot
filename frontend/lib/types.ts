/**
 * types.ts — TypeScript mirror of backend/app/models/schemas.py.
 *
 * These types are hand-kept in lockstep with the Pydantic models so the
 * request/response shapes can't silently drift. If a field changes in
 * schemas.py, it changes here. The `as const` enums mirror the string
 * values Pydantic accepts (the enum *values*, not the Python identifiers).
 */

// DietType — matches DietType enum values in schemas.py.
export const DIET_TYPES = [
  "Non-Vegetarian",
  "Ovo-Vegetarian",
  "Vegetarian",
] as const;
export type DietType = (typeof DIET_TYPES)[number];

// MedicalCondition — matches MedicalCondition enum values in schemas.py.
export const MEDICAL_CONDITIONS = [
  "Low iron",
  "Hypertension",
  "Diabetes",
] as const;
export type MedicalCondition = (typeof MEDICAL_CONDITIONS)[number];

// ResponseType — the discriminator the frontend branches on. Never parse
// answer text; branch on this (schemas.py: ChatResponse docstring).
export type ResponseType = "answer" | "emergency" | "out_of_scope" | "no_results";

// UserProfile — matches UserProfile in schemas.py, including field bounds.
export interface UserProfile {
  name: string; // 1–100 chars
  age: number; // 10–60
  pregnancy_week: number; // 1–45
  diet_type: DietType;
  weight_kg: number; // > 0, ≤ 300
  height_cm: number; // > 0, ≤ 250
  medical_conditions: MedicalCondition[];
}

// Source — a citation surfaced alongside an answer.
export interface Source {
  org_display_name: string;
  doc_title: string;
  page: number;
  year_published: number;
}

// ChatRequest / ChatResponse — the /chat endpoint contract.
export interface ChatRequest {
  message: string; // 1–1000 chars
  user_profile: UserProfile;
}

export interface ChatResponse {
  response_type: ResponseType;
  answer: string;
  sources: Source[];
}
