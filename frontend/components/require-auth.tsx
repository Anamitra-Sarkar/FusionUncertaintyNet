"use client";
import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { onAuthStateChanged, type User } from "firebase/auth";
import { auth } from "@/lib/firebase";

/**
 * Route-level auth gate.
 * - loading  -> full-page skeleton (protected UI never mounts)
 * - anon     -> redirect to /login?next=<current path>
 * - authed   -> render children
 */
export default function RequireAuth({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => onAuthStateChanged(auth, (u) => { setUser(u); setLoading(false); }), []);

  useEffect(() => {
    if (!loading && !user) {
      router.replace(`/login?next=${encodeURIComponent(pathname)}`);
    }
  }, [loading, user, pathname, router]);

  if (loading || !user) {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 text-muted">
        <div className="w-10 h-10 rounded-xl bg-accent/10 text-accent flex items-center justify-center font-serif animate-pulse">◈</div>
        <div className="text-sm animate-pulse">{loading ? "Checking your session…" : "Redirecting to sign-in…"}</div>
      </div>
    );
  }
  return <>{children}</>;
}
