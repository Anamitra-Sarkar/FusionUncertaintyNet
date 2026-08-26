"use client";
import { useEffect, useState } from "react";
import { history } from "@/lib/api";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import RequireAuth from "@/components/require-auth";

const fmt = (s?: string) => { try { return s ? new Date(s).toLocaleString(undefined,{dateStyle:"medium",timeStyle:"short"}) : ""; } catch { return s || ""; } };

function HistoryInner(){
  const [items,setItems]=useState<any[]>([]); const [err,setErr]=useState("");
  useEffect(()=>{
    (async () => {
      try{ const r=await history(); setItems(r.items||[]); } catch(e:any){ setErr(e.message); }
    })();
  },[]);
  return (
    <div className="space-y-6">
      <h1 className="font-serif text-3xl">History</h1>
      <div className="text-sm text-muted">Every analysis you have run, private to your account.</div>
      {err && <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
      <div className="grid gap-3">
        {items.length===0 && <Card><CardContent className="p-8 text-sm text-muted">No predictions yet — run one in Dashboard.</CardContent></Card>}
        {items.map((it:any)=>(
          <Card key={it.id}>
            <CardHeader><div className="flex flex-wrap justify-between gap-x-3 gap-y-1"><span className="font-mono text-xs">{it.id.slice(0,16)}…</span><span className="text-xs text-muted">{fmt(it.created_at)}</span></div></CardHeader>
            <CardContent className="text-sm">
              <div className="font-mono text-xs break-all bg-sand p-2 rounded-xl border border-line">{it.sequence?.slice(0,120)}…</div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs"><span>Len {it.length}</span><span>Q {Number(it.global_quality||0).toFixed(1)}</span><span>U {Number(it.global_uncertainty||0).toFixed(2)}</span><span>Gates {it.gates?.map((g:number)=>g.toFixed(2)).join("·")}</span></div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}


export default function HistoryPage() {
  return (
    <RequireAuth>
      <HistoryInner />
    </RequireAuth>
  );
}
