"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import AuthCta from "@/components/auth-cta";
import { RibbonArt } from "@/components/art";

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
            <span className="w-2 h-2 rounded-full bg-accent" /> Release-gated structural-biology research
          </div>
          <h1 className="font-serif text-[28px] leading-tight sm:text-4xl sm:leading-[1.05] md:text-5xl leading-[1.05] mt-4 tracking-tight">
            Evidence before a <span className="text-accent italic">score.</span>
          </h1>
          <p className="text-muted mt-4 leading-relaxed">
            FusionUncertaintyNet is a research framework for studying protein-structure reliability. It keeps inference unavailable until a reviewed, immutable model artifact and its evaluation evidence are registered.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 mt-6">
            <Link href="/login" className="px-6 py-3 rounded-full bg-accent text-white hover:bg-teal-700 transition text-center w-full sm:w-auto">Sign in to review access</Link>
            <a href="#science" className="px-6 py-3 rounded-full border border-line bg-card hover:bg-sand transition text-center w-full sm:w-auto">Review release criteria</a>
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-2 mt-6 sm:mt-8 text-xs text-muted">
            <span>◆ Immutable artifact required</span><span>◆ Independent review required</span><span>◆ Explicit abstention when unavailable</span>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.1 }} className="bg-card border border-line rounded-2xl p-5 sm:p-6 shadow-sm overflow-hidden">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium tracking-widest text-muted">MODEL RELEASE STATUS</div>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-accent2/10 text-accent2 font-medium">No approved artifact</span>
          </div>
          <div className="mt-2 -mx-5 sm:-mx-6 border-b border-line">
            <RibbonArt className="w-full h-36 sm:h-44" id="hero" />
          </div>
          <div className="mt-4 bg-sand rounded-xl p-4 border border-line">
            <p className="text-sm font-medium text-ink">Inference is not available.</p>
            <p className="mt-2 text-xs sm:text-sm text-muted leading-relaxed">
              No approved immutable model artifact is configured for this research release. The service will not display generated quality scores, uncertainty values, gating weights, or an example sequence until independent release criteria are met.
            </p>
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2.5 sm:gap-3 text-xs">
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Artifact</div><div className="mt-1 font-medium">Not approved</div></div>
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Readiness</div><div className="mt-1 font-medium">Abstaining</div></div>
              <div className="bg-white rounded-lg p-3 border border-line"><div className="text-muted">Access</div><div className="mt-1 font-medium">Identity required</div></div>
            </div>
          </div>
        </motion.div>
      </section>

      <section id="science" className="grid md:grid-cols-3 gap-6">
        {[
          { t: "Provenance first", d: "Any future release must bind a model revision, artifact hash, intended research scope, and the exact supporting evaluation record." },
          { t: "Independent review", d: "Release requires documented calibration and out-of-distribution evaluation plus independent review; implementation details alone are not evidence of performance." },
          { t: "Fail closed", d: "When a reviewed artifact or service dependency is unavailable, the platform abstains instead of generating a structural-quality result." },
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
