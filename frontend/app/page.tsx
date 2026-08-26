"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import AuthCta from "@/components/auth-cta";

export default function Home() {
  return (
    <div className="space-y-12 sm:space-y-16">
      <section className="relative grid md:grid-cols-2 gap-10 items-center pt-2 sm:pt-6">
        <div aria-hidden className="pointer-events-none absolute -top-24 -left-24 w-[420px] h-[420px] rounded-full opacity-60 blur-3xl"
             style={{background:"radial-gradient(closest-side, rgba(15,118,110,0.14), transparent)"}} />
        <div aria-hidden className="pointer-events-none absolute -bottom-32 right-0 w-[360px] h-[360px] rounded-full opacity-50 blur-3xl"
             style={{background:"radial-gradient(closest-side, rgba(232,93,63,0.10), transparent)"}} />
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="inline-flex items-center gap-2 text-xs border border-line rounded-full px-3 py-1 bg-card">
            <span className="w-2 h-2 rounded-full bg-accent animate-pulse" /> Adaptive Multi-PLM · Evidential Gamma
          </div>
          <h1 className="font-serif text-[28px] leading-tight sm:text-4xl sm:leading-[1.05] md:text-5xl leading-[1.05] mt-4 tracking-tight">
            Reliability you can <span className="text-accent italic">trust</span>, not just a score.
          </h1>
          <p className="text-muted mt-4 leading-relaxed">
            FusionUncertaintyNet fuses ESM-2, ProtT5 and AlphaFold priors with a learned gating network, then predicts per-residue quality via Gamma evidential heads — separating aleatoric (disorder) from epistemic (out-of-distribution) uncertainty.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 mt-6">
            <Link href="/dashboard" className="px-6 py-3 rounded-full bg-accent text-white hover:bg-teal-700 transition text-center w-full sm:w-auto">Run Prediction</Link>
            <a href="#science" className="px-6 py-3 rounded-full border border-line bg-card hover:bg-sand transition text-center w-full sm:w-auto">How it works</a>
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-2 mt-6 sm:mt-8 text-xs text-muted">
            <span>◆ 501k AFdb training</span><span>◆ CASP16 / CAMEO blind</span><span>◆ ECE + Ramachandran</span>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.1 }} className="bg-card border border-line rounded-2xl p-6 shadow-sm">
          <div className="text-xs font-medium tracking-widest text-muted">LIVE EXAMPLE</div>
          <div className="mt-3 font-mono text-xs sm:text-sm bg-sand rounded-xl p-4 border border-line overflow-hidden">
            <div className="text-muted text-[10px] sm:text-xs break-all leading-relaxed">MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDNPLDAELLAQLGVD</div>
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2.5 sm:gap-3 text-xs">
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Global quality</div><div className="text-xl font-serif">84.2<span className="text-sm text-muted">/100</span></div></div>
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Uncertainty</div><div className="text-xl font-serif">0.42</div></div>
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Gating</div><div className="text-xs leading-tight">ESM 0.42 · ProtT5 0.33 · AF 0.25</div></div>
            </div>
            <div className="mt-4 h-2 bg-line rounded-full overflow-hidden flex">
              <div className="h-full bg-accent" style={{ width: "84%" }} />
              <div className="h-full bg-accent2/40" style={{ width: "16%" }} />
            </div>
            <div className="text-[11px] text-muted mt-1 flex justify-between"><span>pLDDT-like</span><span>aleatoric vs epistemic shaded</span></div>
          </div>
        </motion.div>
      </section>

      <section id="science" className="grid md:grid-cols-3 gap-6">
        {[
          { t: "Representation", d: "ESM-2 650M (evolution) + ProtT5-XL (motifs) + AF pLDDT/φψ/PAE (structure prior). Only the fusion core is trained — fast, focused, efficient." },
          { t: "Adaptive Fusion", d: "Gating MLP on length, charged_fraction, disorder (SETH). Learns to down-weight AF priors for disordered proteins, up-weight ESM for long chains." },
          { t: "Evidential Gamma", d: "Dual head: μ & (k,θ). Mean kθ = quality, var kθ² = aleatoric, 1/k = epistemic. Loss MSE + Gamma NLL + Ramachandran. Calibrated via ECE/Brier." },
        ].map((c) => (
          <div key={c.t} className="bg-card border border-line rounded-2xl p-6">
            <div className="font-serif text-lg">{c.t}</div>
            <div className="text-sm text-muted mt-2 leading-relaxed">{c.d}</div>
          </div>
        ))}
      </section>

      <section className="bg-ink text-paper rounded-2xl p-8 flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <div className="font-serif text-2xl">Your private protein workspace.</div>
          <div className="text-sm text-white/60 mt-1">Every analysis lives in your own private workspace — full history and AI-powered explanations included.</div>
        </div>
        <AuthCta />
      </section>
    </div>
  );
}
