"use client";
import { useState } from "react";
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

  const doEmail = async () => {
    setErr("");
    try {
      if (mode==="login") await signInWithEmailAndPassword(auth, email, pwd);
      else await createUserWithEmailAndPassword(auth, email, pwd);
      router.push("/dashboard");
    } catch (e:any) { setErr(e.message); }
  };
  const doGoogle = async () => {
    setErr("");
    try { await signInWithPopup(auth, googleProvider); router.push("/dashboard"); } catch(e:any){ setErr(e.message); }
  };

  return (
    <div className="max-w-md mx-auto pt-10">
      <Card>
        <CardHeader>
          <div className="font-serif text-2xl">Welcome back</div>
          <div className="text-sm text-muted">Sign in to run calibrated predictions. Uses <span className="font-mono text-xs">cabbage-guard</span> — same account works across our 35 apps (fusion_* namespaced, no collision).</div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 text-sm">
            <button onClick={()=>setMode("login")} className={`flex-1 py-2 rounded-full border ${mode==="login"?"bg-ink text-white border-ink":"bg-card border-line"}`}>Login</button>
            <button onClick={()=>setMode("signup")} className={`flex-1 py-2 rounded-full border ${mode==="signup"?"bg-ink text-white border-ink":"bg-card border-line"}`}>Create account</button>
          </div>
          <Input placeholder="email" value={email} onChange={e=>setEmail(e.target.value)} />
          <Input placeholder="password" type="password" value={pwd} onChange={e=>setPwd(e.target.value)} />
          {err && <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
          <Button onClick={doEmail} className="w-full">{mode==="login"?"Sign in":"Create account"}</Button>
          <div className="relative py-2"><div className="absolute inset-0 flex items-center"><div className="w-full border-t border-line" /></div><div className="relative flex justify-center"><span className="bg-card px-3 text-xs text-muted">or</span></div></div>
          <Button variant="outline" onClick={doGoogle} className="w-full">Continue with Google</Button>
          <div className="text-xs text-muted">Login is mandatory — predictions are private to your UID, stored in <span className="font-mono">fusion_predictions</span>.</div>
        </CardContent>
      </Card>
    </div>
  );
}
