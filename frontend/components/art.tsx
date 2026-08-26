"use client";

/**
 * Bespoke brand artwork (hand-authored SVG, no external assets).
 * Abstract protein-ribbon helix with an uncertainty envelope —
 * matches the paper palette: teal #0F766E, terracotta #E85D3F, sand/paper.
 */

export function RibbonArt({ className = "", id = "ra" }: { className?: string; id?: string }) {
  return (
    <svg viewBox="0 0 560 420" fill="none" className={className} role="img" aria-label="Abstract protein ribbon illustration">
      <defs>
        <linearGradient id={`${id}-g1`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#0F766E" stopOpacity="0.9" />
          <stop offset="55%" stopColor="#2DD4BF" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#E85D3F" stopOpacity="0.75" />
        </linearGradient>
        <linearGradient id={`${id}-g2`} x1="1" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#E85D3F" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#0F766E" stopOpacity="0.15" />
        </linearGradient>
        <radialGradient id={`${id}-glow`} cx="50%" cy="42%" r="60%">
          <stop offset="0%" stopColor="#0F766E" stopOpacity="0.10" />
          <stop offset="100%" stopColor="#0F766E" stopOpacity="0" />
        </radialGradient>
      </defs>

      <rect width="560" height="420" fill={`url(#${id}-glow)`} />

      {/* uncertainty envelope (dotted band) */}
      <path d="M40 300 C 140 180, 220 380, 320 240 S 480 140, 530 210"
            stroke="#C2B8A8" strokeWidth="26" strokeDasharray="2 14" strokeLinecap="round" opacity="0.55" />

      {/* main ribbon strand */}
      <path d="M40 290 C 140 170, 220 370, 320 230 S 480 130, 530 200"
            stroke={`url(#${id}-g1)`} strokeWidth="13" strokeLinecap="round" />
      {/* companion strand */}
      <path d="M70 330 C 160 230, 240 400, 340 270 S 490 180, 535 245"
            stroke={`url(#${id}-g2)`} strokeWidth="8" strokeLinecap="round" opacity="0.85" />

      {/* residue nodes along ribbon */}
      {[[92,258],[150,232],[208,296],[266,318],[322,244],[382,196],[444,166],[497,182]].map(([x,y],i)=>(
        <circle key={i} cx={x} cy={y} r={i%3===0?7:5}
                fill={i%2? "#E85D3F":"#0F766E"} opacity={0.85}>
          <animate attributeName="opacity" values="0.85;0.45;0.85" dur={`${2.6+i*0.3}s`} repeatCount="indefinite"/>
        </circle>
      ))}

      {/* high-uncertainty callout ring */}
      <circle cx="208" cy="296" r="16" stroke="#E85D3F" strokeWidth="2" strokeDasharray="4 5" opacity="0.8">
        <animateTransform attributeName="transform" type="rotate" from="0 208 296" to="360 208 296" dur="12s" repeatCount="indefinite"/>
      </circle>

      {/* baseline grid ticks */}
      {[80,160,240,320,400,480].map(x=>(
        <line key={x} x1={x} y1="396" x2={x} y2="388" stroke="#C2B8A8" strokeWidth="2" strokeLinecap="round"/>
      ))}
    </svg>
  );
}

/** Compact variant for cards / login panel */
export function MoleculeMark({ className = "", id = "mm" }: { className?: string; id?: string }) {
  return (
    <svg viewBox="0 0 200 120" fill="none" className={className} aria-hidden>
      <defs>
        <linearGradient id={`${id}-m`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#0F766E" /><stop offset="100%" stopColor="#E85D3F" />
        </linearGradient>
      </defs>
      <path d="M12 88 C 48 34, 84 108, 120 56 S 172 28, 190 52"
            stroke={`url(#${id}-m)`} strokeWidth="7" strokeLinecap="round" />
      <path d="M20 100 C 56 52, 92 118, 128 72 S 174 46, 192 64"
            stroke="#C2B8A8" strokeWidth="4" strokeLinecap="round" strokeDasharray="1 8" opacity="0.8"/>
      <circle cx="62" cy="62" r="5" fill="#0F766E"/><circle cx="122" cy="58" r="5" fill="#E85D3F"/>
    </svg>
  );
}
