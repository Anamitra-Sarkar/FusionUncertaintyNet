"use client";
import { useEffect, useState } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import { history } from "@/lib/api";
import { Card, CardHeader, CardContent } from "@/components/ui/card";

export default function HistoryPage(){
  const [items,setItems]=useState<any[]>([]); const [err,setErr]=useState(""); const router=useRouter();
  useEffect(()=>{
    const unsub=onAuthStateChanged(auth, async (u)=>{
      if(!u) router.push("/login");
      else {
        try{ const r=await history(); setItems(r.items||[]); } catch(e:any){ setErr(e.message); }
      }
    });
    return ()=>unsub();
  },[router]);
  return (
    <div className="space-y-6">
      <h1 className="font-serif text-3xl">History</h1>
      <div className="text-sm text-muted">Stored in <span className="font-mono">fusion_predictions</span> on <span className="font-mono">cabbage-guard</span>, isolated per UID. Auto-expires per retention policy.</div>
      {err && <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
      <div className="grid gap-3">
        {items.length===0 && <Card><CardContent className="p-8 text-sm text-muted">No predictions yet — run one in Dashboard.</CardContent></Card>}
        {items.map((it:any)=>(
          <Card key={it.id}>
            <CardHeader><div className="flex justify-between"><span className="font-mono text-xs">{it.id.slice(0,16)}…</span><span className="text-xs text-muted">{it.created_at || ""}</span></div></CardHeader>
            <CardContent className="text-sm">
              <div className="font-mono text-xs break-all bg-sand p-2 rounded-xl border border-line">{it.sequence?.slice(0,120)}…</div>
              <div className="flex gap-4 mt-2 text-xs"><span>Len {it.length}</span><span>Q {Number(it.global_quality||0).toFixed(1)}</span><span>U {Number(it.global_uncertainty||0).toFixed(2)}</span><span>Gates {it.gates?.map((g:number)=>g.toFixed(2)).join("·")}</span></div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
