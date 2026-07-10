/**
 * Wordmark — the Poshan Saathi identity. Shows the name in Devanagari
 * alongside the Latin transliteration, per the project's "keep it culturally
 * familiar" design principle. The mark is a soft rose lotus-ish glyph.
 */
export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div
        className={`grid place-items-center rounded-2xl bg-rose-600 text-sand-50 shadow-sm ${
          compact ? "h-8 w-8 text-lg" : "h-11 w-11 text-2xl"
        }`}
        aria-hidden
      >
        ◕
      </div>
      <div className="leading-tight">
        <div
          className={`font-serif font-semibold tracking-tight text-sand-800 ${
            compact ? "text-base" : "text-xl"
          }`}
        >
          Poshan Saathi
        </div>
        {!compact && (
          <div className="text-sm text-sand-500">
            पोषण साथी · nutrition companion
          </div>
        )}
      </div>
    </div>
  );
}
