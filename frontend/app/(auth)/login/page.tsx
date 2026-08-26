"use client";
import { useState, useEffect } from "react";
import { auth, googleProvider } from "@/lib/firebase";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithPopup, onAuthStateChanged } from "firebase/auth";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [pwd, setPwd] = useState("");
  const [mode, setMode] = useState<"login"|"signup">("login");
  const [err, setErr] = useState("");
  const router = useRouter();

  const [next, setNext] = useState("/dashboard");
  useEffect(() => {
    const n = new URLSearchParams(window.location.search).get("next");
    if (n && n.startsWith("/")) setNext(n);
    return onAuthStateChanged(auth, (u) => { if (u) router.replace(next); });
  }, [router, next]);

  const doEmail = async () => {
    setErr("");
    try {
      if (mode==="login") await signInWithEmailAndPassword(auth, email, pwd);
      else await createUserWithEmailAndPassword(auth, email, pwd);
      router.replace(next);
    } catch (e:any) { setErr(e.message); }
  };
  const doGoogle = async () => {
    setErr("");
    try { await signInWithPopup(auth, googleProvider); router.replace(next); } catch(e:any){ setErr(e.message); }
  };

  return (
    <div className="max-w-md mx-auto px-4 pt-6 sm:pt-10">
      <Card>
        <CardHeader>
          <div className="font-serif text-2xl">Welcome back</div>
          <div className="text-sm text-muted">Run calibrated protein-reliability predictions in your own private workspace.</div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 text-sm">
            <button onClick={()=>setMode("login")} className={`flex-1 py-2 rounded-full border ${mode==="login"?"bg-ink text-white border-ink":"bg-card border-line"}`}>Login</button>
            <button onClick={()=>setMode("signup")} className={`flex-1 py-2 rounded-full border ${mode==="signup"?"bg-ink text-white border-ink":"bg-card border-line"}`}>Create account</button>
          </div>
          <Input placeholder="email" value={email} onChange={e=>setEmail(e.target.value)} />
          <Input placeholder="password" type="password" value={pwd} onChange={e=>setPwd(e.target.value)} />
          {typeof window!=="undefined" && window.location.search.includes("next=") && !err && (
            <div className="text-sm text-accent bg-teal-50 border border-teal-200 rounded-xl p-3">Please sign in to continue to that page.</div>
          )}
          {err && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
          <Button onClick={doEmail} className="w-full">{mode==="login"?"Sign in":"Create account"}</Button>
          <div className="relative py-2"><div className="absolute inset-0 flex items-center"><div className="w-full border-t border-line" /></div><div className="relative flex justify-center"><span className="bg-card px-3 text-xs text-muted">or</span></div></div>
          <Button variant="outline" onClick={doGoogle} className="w-full">Continue with Google</Button>
          <ul className="pt-1 space-y-2 text-xs text-muted">
            {["Private by default — only you see your analyses","Instant analysis history","One-click AI explanations"].map(f=>(
              <li key={f} className="flex items-center gap-2"><span className="w-4 h-4 rounded-full bg-accent/10 text-accent flex items-center justify-center text-[10px]">✓</span>{f}</li>
            ))}
          </ul>
          <div className="text-xs text-muted">Private by default — only you see your analyses.</div>
        </CardContent>
      </Card>
    </div>
  );
}
