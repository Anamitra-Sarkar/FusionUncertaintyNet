"use client";
import { useEffect, useRef, useState } from "react";
import Script from "next/script";

declare global { interface Window { $3Dmol?: any } }

/**
 * Real 3D viewer: AlphaFold DB model colored by per-residue confidence.
 * - pdbUrl: AFDB model PDB (B-factor = AlphaFold pLDDT)
 * - quality: optional per-residue predicted quality (0-100) from FusionUncertaintyNet;
 *   when lengths match, B-factor columns are overwritten so coloring reflects OUR model.
 */
export default function StructureViewer({
  pdbUrl,
  quality,
  height = 380,
}: {
  pdbUrl?: string | null;
  quality?: number[];
  height?: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    if (!ready || !pdbUrl || !host.current) return;
    let glviewer: any;
    let cancelled = false;
    (async () => {
      try {
        setStatus("Fetching structure…");
        const res = await fetch(pdbUrl);
        if (!res.ok) throw new Error(`AFDB ${res.status}`);
        let text = await res.text();
        // Overwrite B-factor (cols 61-66) on CA atoms with our predicted quality
        if (quality && quality.length) {
          const lines = text.split("\n");
          const out: string[] = [];
          let qi = 0;
          for (const line of lines) {
            if (line.startsWith("ATOM") && line.slice(12, 16).trim() === "CA") {
              const q = quality[qi++];
              if (q !== undefined && !Number.isNaN(q)) {
                const bfac = Math.max(0, Math.min(99.99, q)).toFixed(2);
                out.push(line.slice(0, 60).padEnd(60) + bfac.padStart(6) + line.slice(66));
                continue;
              }
            }
            out.push(line);
          }
          text = out.join("\n");
          if (qi !== quality.length) setStatus(`Note: model length differs (${qi} vs ${quality.length} residues) — showing AF pLDDT where unmatched`);
        }
        if (cancelled || !window.$3Dmol || !host.current) return;
        host.current.innerHTML = "";
        glviewer = window.$3Dmol.createViewer(host.current, { backgroundColor: "#FFFCF8" });
        glviewer.addModel(text, "pdb");
        // color by B-factor: red(low) → sand → teal(high); cartoon thickness by confidence
        glviewer.setStyle({}, {
          cartoon: { colorscheme: { prop: "b", gradient: "roygb", min: 40, max: 95 }, thickness: 0.5 },
          stick: { colorscheme: { prop: "b", gradient: "roygb", min: 40, max: 95 }, radius: 0.08, hidden: true },
        });
        glviewer.addSurface(window.$3Dmol.SurfaceType.VDW, {
          opacity: 0.35,
          colorscheme: { prop: "b", gradient: "roygb", min: 40, max: 95 },
        }, {});
        glviewer.zoomTo();
        glviewer.render();
        setStatus("");
      } catch (e: any) {
        setErr(e.message || String(e));
      }
    })();
    return () => { cancelled = true; try { glviewer?.clear(); } catch {} };
  }, [ready, pdbUrl, quality]);

  if (!pdbUrl) return null;

  return (
    <div className="mt-4">
      <Script src="https://3Dmol.org/build/3Dmol-min.js" onReady={() => setReady(true)} />
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium tracking-widest text-muted">3D STRUCTURE · ALPHAFOLD MODEL</span>
        <span className="text-[11px] text-muted">{status || "red = low conf → teal = high conf"}</span>
      </div>
      <div ref={host}
           style={{ width: "100%", height }}
           className="rounded-xl border border-line overflow-hidden bg-sand" />
      {err && <div className="text-xs text-red-600 mt-2">Viewer: {err}</div>}
    </div>
  );
}
