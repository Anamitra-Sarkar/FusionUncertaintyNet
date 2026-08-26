"use client";
import { useEffect, useState } from "react";
import { predict, explain } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import StructureViewer from "@/components/viewer";
import RichText from "@/components/rich-text";
import { MoleculeMark } from "@/components/art";
import RequireAuth from "@/components/require-auth";

function DashboardInner() {
  const [seq, setSeq] = useState("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDNPLDAELLAQLGVD");
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState("");
  const [explanation, setExplanation] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [acc, setAcc] = useState("");

  const run = async () => {
    setLoading(true); setErr(""); setRes(null); setExplanation(""); setAiLoading(false);
    try {
      const cleaned = seq.replace(/[^ACDEFGHIKLMNPQRSTVWY]/gi, "").toUpperCase();
      if (cleaned.length < 10) throw new Error("Sequence too short (need ≥10)");
      if (acc && !/^([OPQ][0-9][A-Z0-9]{3}[0-9])|([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$/.test(acc.trim()))
        throw new Error("Invalid UniProt accession");
      const data = await predict({ sequence: cleaned });
      setRes(data);
    } catch (e:any) { setErr(e.message); } finally { setLoading(false); }
  };

  const doExplain = async () => {
    if (!res) return;
    setAiLoading(true); setExplanation("");
    try {
      const r = await explain({ sequence: res.sequence, global_quality: res.global_quality, global_uncertainty: res.global_uncertainty, gates: res.gates });
      setExplanation(r.explanation);
    } catch(e:any){ setExplanation(""); setErr("AI explanation unavailable — try again in a moment."); }
    finally{ setAiLoading(false); }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-serif text-3xl">Predict Reliability</h1>
        <div className="text-xs text-muted">Calibrated multi-model AI engine</div>
      </div>

      <Card>
        <CardHeader>
          <div className="font-medium">Amino acid sequence (FAST A, 10–1022 aa)</div>
          <div className="text-xs text-muted">Paste raw sequence or FASTA. We auto-strip headers and non-standard AAs (B/Z/X → A).</div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea rows={6} value={seq} onChange={e=>setSeq(e.target.value)} placeholder="MKTAY..." />
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <Button onClick={run} disabled={loading}
                    className="w-full sm:w-auto justify-center">{loading?"Running…":"Run FusionUncertaintyNet"}</Button>
            <span className="text-xs text-muted order-none">{seq.replace(/[^A-Za-z]/g,"").length} aa</span>
            <input value={acc} onChange={e=>setAcc(e.target.value)} inputMode="text"
                   placeholder="UniProt accession (optional — enables 3D view)"
                   className="w-full sm:w-64 sm:ml-auto rounded-full border border-line bg-card px-4 py-2.5 text-sm focus:outline-none focus:border-accent" />
          </div>
          {err && <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-xl p-3">{err}</div>}
        </CardContent>
      </Card>

      {res && (
        <div className="grid md:grid-cols-3 gap-4 sm:gap-6 [&>*]:min-w-0">
          <Card className="md:col-span-2 max-w-full overflow-hidden">
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2"><span className="font-serif text-xl">Per-residue Quality & Uncertainty</span><span className="text-xs border border-line rounded-full px-3 py-1 bg-sand">{res.length} residues · outliers {res.ramachandran_outliers}</span></div>
              <div className="text-xs text-muted">Gates: ESM {res.gates?.[0]?.toFixed(2)} · ProtT5 {res.gates?.[1]?.toFixed(2)} · AF {res.gates?.[2]?.toFixed(2)} · Global {res.global_quality.toFixed(1)}/100 · U {res.global_uncertainty.toFixed(2)}</div>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <span className="sm:hidden absolute right-2 top-1 z-10 text-[10px] text-muted bg-paper/80 rounded-full px-2 py-0.5 border border-line">swipe →</span>
                <div className="h-44 sm:h-48 scroll-x max-w-full border-x border-line rounded-xl bg-sand p-2">
                <div className="flex items-end gap-[2px] h-full" style={{ minWidth: res.residues.length*6 }}>
                  {res.residues.map((r:any)=>(
                    <div key={r.index} className="flex flex-col items-center gap-1" style={{ width: 6 }}>
                      <div className="w-full rounded-sm" title={`${r.aa}${r.index}: Q${r.pred_quality.toFixed(1)} U${r.total_unc.toFixed(2)}`} style={{ height: `${Math.max(4, r.pred_quality*0.8)}px`, background: r.total_unc>1.5 ? "#E85D3F" : r.pred_quality>70 ? "#0F766E" : "#C2B8A8" }} />
                    </div>
                  ))}
                </div>
              </div></div>
              <div className="text-[11px] text-muted mt-2 flex justify-between"><span>Low → High confidence (teal)</span><span>Red = high total uncertainty</span></div>
              <div className="mt-4 max-h-72 sm:max-h-64 scroll-x max-w-full border border-line rounded-xl">
                <table className="w-full min-w-[560px] text-xs">
                  <thead className="bg-sand sticky top-0"><tr><th className="p-2 text-left sticky left-0 bg-sand">#</th><th className="p-2">AA</th><th className="p-2">Q</th><th className="p-2">Ale</th><th className="p-2">Epi</th><th className="p-2">TotU</th></tr></thead>
                  <tbody>{res.residues.slice(0,200).map((r:any)=><tr key={r.index} className="border-t border-line"><td className="p-1.5">{r.index}</td><td className="p-1.5 font-mono">{r.aa}</td><td className="p-1.5">{r.pred_quality.toFixed(1)}</td><td className="p-1.5">{r.aleatoric.toFixed(2)}</td><td className="p-1.5">{r.epistemic.toFixed(3)}</td><td className="p-1.5">{r.total_unc.toFixed(2)}</td></tr>)}</tbody>
                </table>
                {res.residues.length>200 && <div className="p-2 text-xs text-muted text-center">Showing 200/{res.residues.length} — download JSON for full</div>}
              </div>
              <div className="flex flex-col sm:flex-row gap-2 mt-4">
                <Button variant="outline" onClick={()=>{const blob=new Blob([JSON.stringify(res,null,2)],{type:"application/json"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=`fusion_${res.length}.json`; a.click();}}className="w-full sm:w-auto justify-center">Download JSON</Button>
                <Button variant="outline" onClick={doExplain}className="w-full sm:w-auto justify-center">AI Explanation</Button>
              </div>
              {(aiLoading || explanation) && (
                <div className="mt-4 rounded-2xl border border-line bg-gradient-to-b from-sand/80 to-card overflow-hidden">
                  <div className="flex items-center justify-between px-4 py-3 border-b border-line bg-card/80">
                    <div className="flex items-center gap-2">
                      <span className="w-6 h-6 rounded-lg bg-accent/10 text-accent flex items-center justify-center text-xs">✦</span>
                      <span className="text-sm font-medium text-ink">AI Explanation</span>
                    </div>
                    <MoleculeMark className="h-7 w-20 opacity-70" id={`ai${res.length}`} />
                  </div>
                  <div className="px-4 py-3">
                    {aiLoading ? (
                      <div className="space-y-2.5 animate-pulse" aria-label="Generating explanation">
                        {[92,100,78,96,60].map((w,i)=>(
                          <div key={i} className="h-3 rounded-full bg-gradient-to-r from-line via-sand to-line" style={{width:`${w}%`}} />
                        ))}
                      </div>
                    ) : (
                      <RichText text={explanation} />
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader><div className="font-medium">Calibration & Physics</div></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="p-3 bg-sand rounded-xl border border-line"><div className="text-xs text-muted">Global quality</div><div className="text-xl sm:text-2xl font-serif">{res.global_quality.toFixed(1)}<span className="text-base text-muted"> /100</span></div><div className="w-full h-2 bg-line rounded-full mt-2"><div className="h-2 bg-accent rounded-full" style={{width: `${res.global_quality}%`}} /></div></div>
              <div className="p-3 bg-card border border-line rounded-xl"><div className="text-xs text-muted">Uncertainty split</div><div className="text-xs mt-1">Aleatoric captures disorder/noise, epistemic captures OOD. High epistemic → consider experimental validation.</div><div className="mt-2 text-xs font-mono">mean ale { (res.residues.reduce((a:any,b:any)=>a+b.aleatoric,0)/res.residues.length).toFixed(2)} · epi {(res.residues.reduce((a:any,b:any)=>a+b.epistemic,0)/res.residues.length).toFixed(3)}</div></div>
              <div className="p-3 bg-card border border-line rounded-xl"><div className="text-xs text-muted">Adaptive gating</div><div className="mt-2 space-y-1 text-xs">{["ESM-2","ProtT5","AF priors"].map((n,i)=><div key={n} className="flex items-center gap-2"><span className="w-16">{n}</span><div className="flex-1 h-2 bg-line rounded-full"><div className="h-2 bg-accent rounded-full" style={{width: `${(res.gates[i]*100).toFixed(0)}%`}} /></div><span className="font-mono">{res.gates[i].toFixed(2)}</span></div>)}</div></div>
              <div className="text-xs text-muted border-t border-line pt-3">Geometry check · {res.ramachandran_outliers} residues flagged</div>
            </CardContent>
          </Card>

          {acc.trim() && (
            <Card className="md:col-span-3">
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-2">
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


export default function Dashboard() {
  return (
    <RequireAuth>
      <DashboardInner />
    </RequireAuth>
  );
}
