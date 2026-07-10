import type { Config } from "tailwindcss";

/**
 * Design tokens for Poshan Saathi.
 *
 * The palette is deliberately soft, warm, and calm — this is a companion for
 * pregnant women, not a developer tool. A single dusty-rose accent (`rose`)
 * carries all interactive emphasis; everything else is a soft, paper-like
 * neutral scale (`sand`) so the accent never competes for attention. Diet
 * semantics reuse the familiar 🔴🟡🟢 hues (`diet.*`).
 *
 * Accessibility note: white text needs a deep-enough fill, so surfaces with
 * white text (primary buttons, the send button, the user's own bubble) use
 * rose-600 (~4.5:1 AA), while the lighter tints (50–400) do the soft, pretty
 * work on light backgrounds — borders, dots, tints, and dark-on-light text.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        rose: {
          50: "#fdf4f6",
          100: "#fae7ed",
          200: "#f4d1dc",
          300: "#eab0c3",
          400: "#dd88a4",
          500: "#cf6b8a", // soft accent (dots, rings, tints)
          600: "#bd5477", // white-text fills (buttons, user bubble) — AA
          700: "#9f4360",
          800: "#82374e",
          900: "#6d3141",
        },
        sand: {
          50: "#fbfaf8",
          100: "#f4f1ec",
          200: "#e9e3d9",
          300: "#d8cfc0",
          400: "#b6ab99",
          500: "#8f8474",
          600: "#6d6355",
          700: "#514a40",
          800: "#38332d",
          900: "#241f1b",
        },
        diet: {
          nonveg: "#e05252",
          ovo: "#e0a92e",
          veg: "#4c9a5a",
        },
      },
      fontFamily: {
        // Nunito — rounded, gentle, warm — for all body/UI text.
        sans: ["var(--font-nunito)", "ui-sans-serif", "system-ui", "sans-serif"],
        // Fraunces — a soft serif with warmth and authority — for the
        // wordmark and headings, so the app feels knowledgeable, not clinical.
        serif: ["var(--font-fraunces)", "ui-serif", "Georgia", "serif"],
      },
      borderRadius: {
        "2xl": "1.125rem",
        "3xl": "1.5rem",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "blink": {
          "0%, 80%, 100%": { opacity: "0.25" },
          "40%": { opacity: "1" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.35s ease-out both",
        "blink": "blink 1.4s infinite both",
      },
    },
  },
  plugins: [],
};

export default config;
