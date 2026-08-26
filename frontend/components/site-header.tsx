"use client";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";

export default function SiteHeader() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  useEffect(() => setOpen(false), [pathname]);
  const links = [
    { href: "/", label: "Overview" },
    { href: "/dashboard", label: "Predict" },
    { href: "/history", label: "History" },
  ];
  return (
    <header className="sticky top-0 z-20 backdrop-blur bg-paper/80 border-b border-line pt-safe">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between gap-3">
        <a href="/" className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 shrink-0 rounded-lg bg-accent flex items-center justify-center text-white font-serif text-lg">◈</div>
          <div className="min-w-0">
            <div className="font-serif text-base sm:text-[18px] leading-none tracking-tight truncate">FusionUncertaintyNet</div>
            <div className="text-[11px] sm:text-xs text-muted -mt-0.5 truncate">Calibrated Protein Reliability</div>
          </div>
        </a>

        {/* desktop nav */}
        <nav className="hidden lg:flex items-center gap-6 text-sm shrink-0">
          {links.map(l => (
            <a key={l.href} href={l.href} className="hover:text-accent py-2">{l.label}</a>
          ))}
          <a href="/login" className="px-5 py-2.5 rounded-full bg-ink text-white hover:bg-black transition">Sign in</a>
        </nav>

        {/* mobile: compact sign-in + hamburger */}
        <div className="flex lg:hidden items-center gap-2 shrink-0">
          <a href="/login"
             className="px-4 py-2 rounded-full bg-ink text-white text-sm hover:bg-black transition">Sign in</a>
          <button
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen(v => !v)}
            className="w-11 h-11 -mr-1 rounded-xl border border-line bg-card flex flex-col items-center justify-center gap-[5px]">
            <span className={`block w-5 h-[2px] bg-ink transition-transform duration-200 ${open ? "translate-y-[7px] rotate-45" : ""}`} />
            <span className={`block w-5 h-[2px] bg-ink transition-opacity duration-200 ${open ? "opacity-0" : ""}`} />
            <span className={`block w-5 h-[2px] bg-ink transition-transform duration-200 ${open ? "-translate-y-[7px] -rotate-45" : ""}`} />
          </button>
        </div>
      </div>

      {/* mobile drawer */}
      <div className={`lg:hidden overflow-hidden transition-[max-height] duration-300 ease-out ${open ? "max-h-72" : "max-h-0"}`}>
        <nav className="px-4 pb-3 flex flex-col divide-y divide-line border-t border-line">
          {links.map(l => (
            <a key={l.href} href={l.href}
               className="py-3.5 text-[15px] font-medium hover:text-accent active:bg-sand/60 px-1">
              {l.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
