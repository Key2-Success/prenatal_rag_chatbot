/**
 * Wordmark — the Poshan Saathi identity. Shows the name in Devanagari
 * alongside the Latin transliteration, per the project's "keep it culturally
 * familiar" design principle.
 *
 * The mark fuses the two ideas at the heart of the app: a pregnant woman in
 * side profile (head on the right, maternal care) whose round belly is an
 * apple (nutrition), complete with a little stem and leaf at the top.
 */
function MotherAppleMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.1}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      {/* head (right), facing left toward the belly */}
      <circle cx="31" cy="11.3" r="4.7" />
      {/* eye */}
      <circle cx="28.9" cy="10.5" r="0.85" fill="currentColor" stroke="none" />
      {/* long, curly mermaid hair: a crown sweep and three wavy locks whose
          ends billow out to the south-east, away from the body — carefree
          and breezy rather than sleek */}
      <path d="M30 6.8 C 31.5 4.2, 35 3.9, 37.6 7.6 C 41.8 8.8, 41 11.4, 44 12.2 C 45.8 12.7, 45.4 14.4, 46.5 15.5" />
      <path d="M37 11.2 C 40.2 12, 39.4 14.2, 42.2 15 C 44 15.5, 43.4 17.2, 44.5 18.5" />
      <path d="M35.4 13.6 C 37.9 14.4, 37.2 16.4, 39.4 17.4 C 41 18.1, 40.6 19.6, 41.5 21" />
      {/* rounded back — the apple's right side */}
      <path d="M31.7 16 C 37 18.8, 38.5 27, 34 34" />
      {/* front: chest into a round apple belly, curving under to meet the back */}
      <path d="M28 15.5 C 26 17.5, 26 20, 26.5 22 C 20 20, 11 23, 11 30.5 C 11 37.5, 18 41, 24.5 40 C 28.5 39.3, 31.5 37.5, 34 34" />
      {/* apple stem + leaf at the top of the belly — the leaf is drawn large
          and open so the outline reads as a leaf rather than filling in */}
      <path d="M25.5 22 V 17.5" />
      <path d="M25.5 19 C 21 12.5, 15 10.5, 12.5 11.5 C 16 15.5, 21 18, 25.5 19" />
    </svg>
  );
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div
        className={`grid place-items-center rounded-2xl bg-rose-600 text-sand-50 shadow-sm ${
          compact ? "h-8 w-8" : "h-11 w-11"
        }`}
        aria-hidden
      >
        <MotherAppleMark className={compact ? "h-6 w-6" : "h-9 w-9"} />
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
