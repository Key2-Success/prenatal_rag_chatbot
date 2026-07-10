import type { Metadata, Viewport } from "next";
import { Nunito, Fraunces } from "next/font/google";
import "./globals.css";

// Body/UI — rounded and gentle.
const nunito = Nunito({
  subsets: ["latin"],
  variable: "--font-nunito",
  display: "swap",
});

// Wordmark/headings — a soft serif for warmth + authority.
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["SOFT", "opsz"],
});

export const metadata: Metadata = {
  title: "Poshan Saathi — Prenatal Nutrition Companion",
  description:
    "A gentle, evidence-based nutrition companion for pregnant women in India.",
};

export const viewport: Viewport = {
  themeColor: "#bd5477",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${nunito.variable} ${fraunces.variable}`}>
      <body>{children}</body>
    </html>
  );
}
