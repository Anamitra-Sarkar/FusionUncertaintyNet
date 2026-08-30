"use client";
import { useState, useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { onAuthStateChanged, signOut, type User } from "firebase/auth";
import { auth } from "@/lib/firebase";

export default function SiteHeader() {
  const [open, setOpen] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    // firebase auth restores session async; track it
    return onAuthStateChanged(auth, (u) => { setUser(u); setReady(true); });
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  const doSignOut = async () => {
    await signOut(auth);
    setUser(null);
    setOpen(false);
    router.push("/");
    router.refresh();
  };

  const links = [
    { href: "/", label: "Overview" },
    { href: "/dashboard", label: "Workspace" },
    { href: "/history", label: "Research record" },
    { href: "/calibration-evidence", label: "Calibration evidence" },
  ];
  const authed = ready && !!user;
  const initial = user?.email?.[0]?.toUpperCase() ?? "?";

  return (
    <header className="sticky top-0 z-20 backdrop-blur bg-paper/80 border-b border-line pt-safe">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between gap-3">
        <a href="/" className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 shrink-0 rounded-lg bg-accent flex items-center justify-center text-white font-serif text-lg">◈</div>
          <div className="min-w-0">
            <div className="font-serif text-base sm:text-[18px] leading-none tracking-tight truncate">FusionUncertaintyNet</div>
            <div className="text-[11px] sm:text-xs text-muted -mt-0.5 truncate">Release-gated research</div>
          </div>
        </a>

        {/* desktop */}
        <nav className="hidden lg:flex items-center gap-6 text-sm shrink-0">
          {links.map(l => (
            <a key={l.href} href={l.href} className="hover:text-accent py-2">{l.label}</a>
          ))}
          {!authed ? (
            <a href="/login" className="px-5 py-2.5 rounded-full bg-ink text-white hover:bg-black transition">Sign in</a>
          ) : (
            <div className="flex items-center gap-2 pl-1 border-l border-line">
              <a href="/history" title={user?.email ?? ""}
                 className="w-9 h-9 rounded-full bg-accent/10 text-accent flex items-center justify-center text-sm font-semibold hover:ring-2 hover:ring-accent/30 transition">
                {initial}
              </a>
              <button onClick={doSignOut}
                      className="px-4 py-2 rounded-full border border-line bg-card hover:bg-sand transition">
                Sign out
              </button>
            </div>
          )}
        </nav>

        {/* mobile cluster */}
        <div className="flex lg:hidden items-center gap-2 shrink-0">
          {!authed && (
            <a href="/login"
               className="px-4 py-2 rounded-full bg-ink text-white text-sm hover:bg-black transition">Sign in</a>
          )}
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
      <div className={`lg:hidden overflow-hidden transition-[max-height] duration-300 ease-out ${open ? "max-h-96" : "max-h-0"}`}>
        <nav className="px-4 pb-3 flex flex-col divide-y divide-line border-t border-line">
          {authed && (
            <div className="py-3 flex items-center gap-3 px-1">
              <span className="w-8 h-8 rounded-full bg-accent/10 text-accent flex items-center justify-center text-sm font-semibold">{initial}</span>
              <span className="text-sm truncate">{user?.email}</span>
            </div>
          )}
          {links.map(l => (
            <a key={l.href} href={l.href}
               className="py-3.5 text-[15px] font-medium hover:text-accent active:bg-sand/60 px-1">
              {l.label}
            </a>
          ))}
          {authed ? (
            <button onClick={doSignOut}
                    className="py-3.5 text-[15px] font-medium text-left text-red-600 active:bg-red-50 px-1">
              Sign out
            </button>
          ) : (
            <a href="/login" className="py-3.5 text-[15px] font-medium text-left px-1">Sign in</a>
          )}
        </nav>
      </div>
    </header>
  );
}
