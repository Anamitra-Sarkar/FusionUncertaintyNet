"use client";
import { useEffect, useState } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import { predict, explain } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import StructureViewer from "@/components/viewer";

export default function Dashboard() {
  const [seq, setSeq] = useState("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDNPLDAELLAQLGVD");
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState("");
  const [explanation, setExplanation] = useState("");
  const [acc, setAcc] = useState("");
  const router = useRouter();

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => { if (!u) router.push("/login"); });
    return () => unsub();
  }, [router]);

  const run = async () => {
    setLoading(true); setErr(""); setRes(null); setExplanation("");
    try {
      const cleaned = seq.replace(/[^ACDEFGHIKLMNPQRSTVWY]/gi, "").toUpperCase();
      if (cleaned.length < 10) throw new Error("Sequence too short (need ≥10)");
      if (acc && !/^[A-NR-Z][0-9][A-Z0-9]{4}[0-9A-Z]$|^([A-NR-Z][0-9][A-Z][A-Z0-9]{3}[0-9])$/.test(acc.trim()))
        throw new Error("Invalid UniProt accession");
      const data = await predict({ sequence: cleaned });
      setRes(data);
    } catch (e:any) { setErr(e.message); } finally { setLoading(false); }
  };

  const doExplain = async () => {
    if (!res) return;
    try {
      const r = await explain({ sequence: res.sequence, global_quality: res.global_quality, global_uncertainty: res.global_uncertainty, gates: res.gates });
      setExplanation(r.explanation);
    } catch(e:any){ setExplanation("Explain failed: "+e.message); }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="font-serif text-3xl">Predict Reliability</h1>
        <div className="text-xs text-muted">P100-optimized inference via HF Spaces</div>
      </div>

      <Card>
        <CardHeader>
          <div className="font-medium">Amino acid sequence (FAST A, 10–1022 aa)</div>
          <div className="text-xs text-muted">Paste raw sequence or FASTA. We auto-strip headers and non-standard AAs (B/Z/X → A).</div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea rows={6} value={seq} onChange={e=>setSeq(e.target.value)} placeholder="MKTAY..." />
          <div className="flex items-center gap-3">
            <Button onClick={run} disabled={loading}>{loading?"Running…":"Run FusionUncertaintyNet"}</Button>
            <span className="text-xs text-muted">{seq.replace(/[^A-Za-z]/g,"").length} aa</span>
            <input value={acc} onChange={e=>setAcc(e.target.value)}
                   placeholder="UniProt acc (optional, for 3D)"
                   className="ml-auto w-56 rounded-full border border-line bg-card px-4 py-2 text-sm focus:outline-none focus:border-accent" />
          </div>
          {err && <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
        </CardContent>
      </Card>

      {res && (
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="md:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between"><span className="font-serif text-xl">Per-residue Quality & Uncertainty</span><span className="text-xs border border-line rounded-full px-3 py-1 bg-sand">{res.length} residues · outliers {res.ramachandran_outliers}</span></div>
              <div className="text-xs text-muted">Gates: ESM {res.gates?.[0]?.toFixed(2)} · ProtT5 {res.gates?.[1]?.toFixed(2)} · AF {res.gates?.[2]?.toFixed(2)} · Global {res.global_quality.toFixed(1)}/100 · U {res.global_uncertainty.toFixed(2)}</div>
            </CardHeader>
            <CardContent>
              <div className="h-48 overflow-x-auto border border-line rounded-xl bg-sand p-2">
                <div className="flex items-end gap-[2px] h-full" style={{ minWidth: res.residues.length*6 }}>
                  {res.residues.map((r:any)=>(
                    <div key={r.index} className="flex flex-col items-center gap-1" style={{ width: 6 }}>
                      <div className="w-full rounded-sm" title={`${r.aa}${r.index}: Q${r.pred_quality.toFixed(1)} U${r.total_unc.toFixed(2)}`} style={{ height: `${Math.max(4, r.pred_quality*0.8)}px`, background: r.total_unc>1.5 ? "#E85D3F" : r.pred_quality>70 ? "#0F766E" : "#C2B8A8" }} />
                    </div>
                  ))}
                </div>
              </div>
              <div className="text-[11px] text-muted mt-2 flex justify-between"><span>Low → High confidence (teal)</span><span>Red = high total uncertainty</span></div>
              <div className="mt-4 max-h-64 overflow-auto border border-line rounded-xl">
                <table className="w-full text-xs">
                  <thead className="bg-sand sticky top-0"><tr><th className="p-2 text-left">#</th><th className="p-2">AA</th><th className="p-2">Q</th><th className="p-2">Ale</th><th className="p-2">Epi</th><th className="p-2">TotU</th></tr></thead>
                  <tbody>{res.residues.slice(0,200).map((r:any)=><tr key={r.index} className="border-t border-line"><td className="p-1.5">{r.index}</td><td className="p-1.5 font-mono">{r.aa}</td><td className="p-1.5">{r.pred_quality.toFixed(1)}</td><td className="p-1.5">{r.aleatoric.toFixed(2)}</td><td className="p-1.5">{r.epistemic.toFixed(3)}</td><td className="p-1.5">{r.total_unc.toFixed(2)}</td></tr>)}</tbody>
                </table>
                {res.residues.length>200 && <div className="p-2 text-xs text-muted text-center">Showing 200/{res.residues.length} — download JSON for full</div>}
              </div>
              <div className="flex gap-2 mt-4">
                <Button variant="outline" onClick={()=>{const blob=new Blob([JSON.stringify(res,null,2)],{type:"application/json"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=`fusion_${res.length}.json`; a.click();}}>Download JSON</Button>
                <Button variant="outline" onClick={doExplain}>Explain with Groq</Button>
              </div>
              {explanation && <div className="mt-4 bg-sand border border-line rounded-xl p-4 text-sm leading-relaxed whitespace-pre-wrap">{explanation}</div>}
            </CardContent>
          </Card>
          <Card>
            <CardHeader><div className="font-medium">Calibration & Physics</div></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="p-3 bg-sand rounded-xl border border-line"><div className="text-xs text-muted">Global quality</div><div className="text-2xl font-serif">{res.global_quality.toFixed(1)}<span className="text-base text-muted"> /100</span></div><div className="w-full h-2 bg-line rounded-full mt-2"><div className="h-2 bg-accent rounded-full" style={{width: `${res.global_quality}%`}} /></div></div>
              <div className="p-3 bg-card border border-line rounded-xl"><div className="text-xs text-muted">Uncertainty split</div><div className="text-xs mt-1">Aleatoric captures disorder/noise, epistemic captures OOD. High epistemic → consider experimental validation.</div><div className="mt-2 text-xs font-mono">mean ale { (res.residues.reduce((a:any,b:any)=>a+b.aleatoric,0)/res.residues.length).toFixed(2)} · epi {(res.residues.reduce((a:any,b:any)=>a+b.epistemic,0)/res.residues.length).toFixed(3)}</div></div>
              <div className="p-3 bg-card border border-line rounded-xl"><div className="text-xs text-muted">Adaptive gating</div><div className="mt-2 space-y-1 text-xs">{["ESM-2","ProtT5","AF priors"].map((n,i)=><div key={n} className="flex items-center gap-2"><span className="w-16">{n}</span><div className="flex-1 h-2 bg-line rounded-full"><div className="h-2 bg-accent rounded-full" style={{width: `${(res.gates[i]*100).toFixed(0)}%`}} /></div><span className="font-mono">{res.gates[i].toFixed(2)}</span></div>)}</div></div>
              <div className="text-xs text-muted border-t border-line pt-3">Ramachandran outliers: {res.ramachandran_outliers} (proxy via uncertainty+φψ penalty) · Model: {res.model_version}</div>
            </CardContent>
          </Card>

          {acc.trim() && (
            <Card className="md:col-span-3">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <span className="font-medium">Structure — colored by FusionUncertaintyNet quality</span>
                  <a className="text-xs text-accent hover:underline"
                     href={`https://alphafold.ebi.ac.uk/api/prediction/${acc.trim()}`} target="_blank" rel="noreferrer">
                    AFDB entry ↗</a>
                </div>
              </CardHeader>
              <CardContent>
                <StructureViewer
                  pdbUrl={`https://fusionuncertaintynet-lite.onrender.com/api/pdb?acc=${acc.trim()}`}
                  quality={res.residues.map((r:any)=>r.pred_quality)}
                />
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
