"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { onAuthStateChanged, type User } from "firebase/auth";
import { auth } from "@/lib/firebase";

/** Landing CTA that respects auth state: sign-in prompt vs workspace shortcut. */
export default function AuthCta() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => onAuthStateChanged(auth, (u) => { setUser(u); setReady(true); }), []);

  if (!ready) return <div className="h-[52px]" />; // reserve height, avoid layout shift

  return user ? (
    <Link href="/dashboard"
          className="inline-block px-6 py-3 rounded-full bg-white text-ink hover:bg-sand transition">
      Open my workspace →
    </Link>
  ) : (
    <Link href="/login"
          className="px-6 py-3 rounded-full bg-white text-ink hover:bg-sand transition">
      Sign in to continue →
    </Link>
  );
}
