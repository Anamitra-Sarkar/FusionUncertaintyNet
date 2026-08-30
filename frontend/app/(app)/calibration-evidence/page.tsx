"use client";
import RequireAuth from "@/components/require-auth";
import { Card, CardHeader, CardContent } from "@/components/ui/card";

function CalibrationEvidenceInner() {
  return (
    <div className="space-y-6">
      {/* Header — matches dashboard/history: serif h1, muted subtitle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="font-serif text-3xl">Calibration Evidence</h1>
        <div className="text-xs text-muted">Real held-out evaluation — not a live-serving claim</div>
      </div>
      <div className="text-sm text-muted leading-relaxed max-w-3xl">
        Both scientific gates from <span className="font-mono text-xs bg-sand border border-line rounded px-1.5 py-0.5">docs/model-card.md</span> release checklist have
        cleared for the first time on a real held-out split — point prediction and uncertainty calibration — documented in{" "}
        <span className="font-mono text-xs bg-sand border border-line rounded px-1.5 py-0.5">docs/interval-calibration-result-2026-08-30.md</span>.
        This page surfaces those real numbers honestly, including what is <em>not</em> yet true in production.
      </div>

      {/* Honest banner — what is NOT yet true */}
      <Card className="border-amber-200 bg-amber-50/70">
        <CardContent className="p-4 sm:p-5">
          <div className="flex gap-3">
            <span className="shrink-0 w-7 h-7 rounded-full bg-amber-100 border border-amber-200 flex items-center justify-center text-amber-700 text-sm">!</span>
            <div className="space-y-1.5 min-w-0">
              <div className="text-sm font-medium text-amber-900">Not yet live in production — evaluation evidence only</div>
              <p className="text-xs sm:text-sm text-amber-900/80 leading-relaxed">
                The live production app does <strong>not</strong> currently apply this isotonic calibration by default. Predictions users see today still
                use the raw uncertainty head (interval calibration error ~0.18). A fitted calibration artifact exists at{" "}
                <span className="font-mono text-[11px] break-all bg-white border border-amber-200 rounded px-1.5 py-0.5">bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree:calibration/calibration-isotonic-v1.json</span>{" "}
                (HTTP 200 verified) and an opt-in serving path on branch{" "}
                <span className="font-mono text-xs bg-white border border-amber-200 rounded px-1.5 py-0.5">interval-calibration-serving-wiring</span> can apply it when{" "}
                <span className="font-mono text-xs bg-white border border-amber-200 rounded px-1.5 py-0.5">CALIBRATION_ARTIFACT_ENABLED=true</span> — but that flag is{" "}
                <strong>not</strong> on in production. This page reports real evaluation evidence; it does not claim predictions you see today are already calibrated.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Headline gates — two stat cards matching dashboard card pattern */}
      <div className="grid md:grid-cols-2 gap-4 sm:gap-6">
        <Card>
          <CardHeader>
            <div className="text-xs tracking-widest text-muted font-medium">POINT-PREDICTION GATE</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="font-serif text-2xl sm:text-3xl">Pearson r = 0.881</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">PASS</span>
            </div>
            <div className="text-xs text-muted mt-1">Bar: &gt;0.7 strong · Held-out 841 proteins / 271,311 residues · checkpoint bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree</div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-xs text-muted leading-relaxed bg-sand rounded-xl p-3 border border-line">
              Training-time val Pearson was 0.8622; post-fix Modal re-eval confirms 0.881 (slightly exceeds). Same checkpoint, encoder_mode=small correctly auto-detected, phi/psi forwarded. Bug-fix, not model, was the prior blocker.
            </div>
          </CardContent>
        </Card>

        <Card className="border-emerald-200">
          <CardHeader>
            <div className="text-xs tracking-widest text-muted font-medium">UNCERTAINTY-CALIBRATION GATE</div>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="font-serif text-2xl sm:text-3xl">ICE = 0.0070</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">PASS · 10× inside bar</span>
            </div>
            <div className="text-xs text-muted mt-1">Bar: &lt;0.07 suggested (&lt;0.05 preferred) · Isotonic PIT recalibration · TEST 411 proteins / 127,240 residues (never used for fitting)</div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="text-xs text-muted leading-relaxed bg-emerald-50/60 rounded-xl p-3 border border-emerald-200">
              Real held-out TEST split the isotonic map never saw. Clears the &lt;0.07 bar by ~10× for the first time this session.
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Progression table — raw → temperature → isotonic */}
      <Card>
        <CardHeader>
          <div className="font-medium">Calibration progression — same checkpoint, real held-out splits</div>
          <div className="text-xs text-muted mt-1">interval_calibration_error (lower is better) vs &lt;0.07 bar. Temperature scalar alone cannot close the gap; isotonic does.</div>
        </CardHeader>
        <CardContent>
          <div className="scroll-x max-w-full border border-line rounded-xl overflow-hidden">
            <table className="w-full min-w-[640px] text-xs sm:text-sm">
              <thead className="bg-sand">
                <tr className="text-left">
                  <th className="p-3 font-medium">Stage</th>
                  <th className="p-3 font-medium">FIT split ICE</th>
                  <th className="p-3 font-medium">TEST split ICE (honest)</th>
                  <th className="p-3 font-medium">Verdict vs &lt;0.07</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line bg-card">
                <tr>
                  <td className="p-3">
                    <div className="font-medium">Raw (no correction)</div>
                    <div className="text-xs text-muted">Normal(pred, var=k·θ²) · 0.181 overall</div>
                  </td>
                  <td className="p-3 font-mono">0.1794</td>
                  <td className="p-3 font-mono">0.1825</td>
                  <td className="p-3"><span className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 text-xs font-medium">FAIL</span></td>
                </tr>
                <tr>
                  <td className="p-3">
                    <div className="font-medium">Temperature scalar (single T)</div>
                    <div className="text-xs text-muted">T refit on same fit split = 0.235 · earlier Modal T=0.21 gave 0.086 on 407/434 split</div>
                  </td>
                  <td className="p-3 font-mono">0.0769 <span className="text-muted">(0.076 on 407-fit)</span></td>
                  <td className="p-3 font-mono">0.0790 <span className="text-muted">· 0.086 earlier run</span></td>
                  <td className="p-3"><span className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 text-xs font-medium">FAIL</span></td>
                </tr>
                <tr className="bg-emerald-50/40">
                  <td className="p-3">
                    <div className="font-medium">Isotonic PIT (level-dependent)</div>
                    <div className="text-xs text-muted">IsotonicRegression on PIT u = Φ((y−pred)/std); monotone nominal→empirical map</div>
                  </td>
                  <td className="p-3 font-mono">0.0000034 <span className="text-muted">(fit — optimistic)</span></td>
                  <td className="p-3 font-mono font-semibold">0.0070</td>
                  <td className="p-3"><span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-medium">PASS</span></td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-muted mt-2 leading-relaxed">
            Raw 0.183 cited in the task summary rounds the Modal 0.181–0.183 range (0.1809 overall, 0.1825 on the isotonic TEST split, 0.1794 on its FIT split). Temperature shows both
            the earlier Modal run (T=0.21, TEST 0.086) and the newer refit on the isotonic split (T=0.235, TEST 0.079) — both still FAIL, confirming a scalar cannot fix shape miscalibration.
            Report TEST error as the honest number; FIT error is optimistic.
          </div>
        </CardContent>
      </Card>

      {/* Coverage detail — why raw/temperature fail */}
      <div className="grid md:grid-cols-2 gap-4 sm:gap-6">
        <Card>
          <CardHeader><div className="font-medium">Why raw fails — systematic over-coverage</div></CardHeader>
          <CardContent className="text-sm space-y-3">
            <p className="text-xs text-muted leading-relaxed">
              Raw intervals are far wider than they need to be at every nominal level (Modal post-fix run, 841 proteins):
            </p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-sand rounded-xl p-3 border border-line"><div className="text-muted">50% nominal → captures</div><div className="font-mono font-medium mt-1">91.8%</div></div>
              <div className="bg-sand rounded-xl p-3 border border-line"><div className="text-muted">80% nominal → captures</div><div className="font-mono font-medium mt-1">97.6%</div></div>
              <div className="bg-sand rounded-xl p-3 border border-line"><div className="text-muted">90% nominal → captures</div><div className="font-mono font-medium mt-1">98.7%</div></div>
              <div className="bg-sand rounded-xl p-3 border border-line"><div className="text-muted">95% nominal → captures</div><div className="font-mono font-medium mt-1">99.2%</div></div>
            </div>
            <p className="text-xs text-muted leading-relaxed">
              Temperature T=0.21 over-corrects the tails: 50%→62.7% (still over), but 90%→81.6% and 95%→84.4% now <em>under</em>-covering. A single global scalar cannot fix a shape error — isotonic&apos;s level-dependent remapping is needed.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><div className="font-medium">Honest caveats — what this does not prove</div></CardHeader>
          <CardContent className="text-xs leading-relaxed space-y-2.5 text-muted">
            <p>
              <strong className="text-ink">Isotonic flexibility & overfit risk:</strong> Isotonic is more flexible than a single scalar and can overfit the calibration map when the FIT split is small.
              With ~430 proteins / ~144k residues (fit split) the 1D monotone map has ample data, but FIT error (0.0000034) is optimistic — report TEST 0.0070 as the honest number.
            </p>
            <p>
              <strong className="text-ink">Normal approximation:</strong> The <span className="font-mono text-[11px] bg-sand border border-line rounded px-1 py-0.5">Normal(pred, var=k·θ²)</span> predictive is itself a model choice; true predictive may be non-Gaussian. Isotonic PIT recalibration corrects marginal CDF shape but not conditional structure.
            </p>
            <p>
              <strong className="text-ink">Serving gap (lossy scalar):</strong> Isotonic is level-dependent (different factor per 50/80/90/95% nominal). Live serving must return a single scalar aleatoric/total_unc, so the opt-in path averages width ratios across levels to get one effective theta factor — lossy. True per-level intervals would need an API change. The 0.0070 is reproduced via level-dependent coverage, not the single-factor scalar.
            </p>
            <p>
              <strong className="text-ink">No new training:</strong> Checkpoint is <span className="font-mono text-[11px]">bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree</span> (encoder_mode=small). Isotonic map was fitted on held-out data only; no retrain.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Artifact & provenance */}
      <Card>
        <CardHeader><div className="font-medium">Artifact & provenance</div></CardHeader>
        <CardContent className="text-sm space-y-3">
          <div className="grid sm:grid-cols-2 gap-3 text-xs">
            <div className="bg-sand rounded-xl p-3 border border-line">
              <div className="text-muted">Calibration artifact (uploaded, HTTP 200 confirmed)</div>
              <div className="font-mono text-[11px] break-all mt-1">bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree:calibration/calibration-isotonic-v1.json</div>
              <div className="text-muted mt-2">Fitted map: 500-point dense table (x=nominal, y=empirical) + metadata (fit date 2026-08-30, split sizes, TEST 0.0070)</div>
            </div>
            <div className="bg-card rounded-xl p-3 border border-line">
              <div className="text-muted">Serving wiring (opt-in, not live)</div>
              <div className="font-mono text-xs mt-1">branch interval-calibration-serving-wiring</div>
              <div className="text-muted mt-1">backend-heavy/fusionuncertaintynet/calibration.py + app/main.py calibrated when <span className="font-mono text-[11px] bg-sand border border-line rounded px-1 py-0.5">CALIBRATION_ARTIFACT_ENABLED=true</span> (default false, byte-identical fallback, fail-closed on error)</div>
              <div className="text-muted mt-2">Real split: 430 fit / 144,071 residues · 411 test / 127,240 residues · md5-salted protein-level, disjoint from val-set hash</div>
            </div>
          </div>
          <div className="text-xs text-muted leading-relaxed border-t border-line pt-3">
            Combined headline: point Pearson 0.881 (PASS, &gt;0.7) + isotonic interval_calibration_error 0.0070 (PASS, &lt;0.07) — both gates now have real held-out evidence for the first time. This alone does not flip <span className="font-mono text-[11px] bg-sand border border-line rounded px-1 py-0.5">MODEL_RELEASE_APPROVED</span>; that requires coordinator/human review and deploying the map alongside the checkpoint at inference time.
          </div>
        </CardContent>
      </Card>

      {/* Source record */}
      <Card>
        <CardContent className="p-4 sm:p-5">
          <div className="text-xs text-muted leading-relaxed">
            <span className="font-medium text-ink">Source record:</span>{" "}
            <span className="font-mono text-xs bg-sand border border-line rounded px-1.5 py-0.5">docs/interval-calibration-result-2026-08-30.md</span>{" "}
            — see top section for point-prediction gate (Pearson 0.881) and the newest <em>Isotonic recalibration result</em> section near the bottom for the full TEST splits, coverage detail, and verbatim caveats. Also{" "}
            <span className="font-mono text-xs bg-sand border border-line rounded px-1.5 py-0.5">docs/promotion-decision-v2-2026-08-28.md §8.2</span> for the &lt;0.07 / &lt;0.05 bar.
            <span className="ml-2 text-ink">Do not interpret this page as a claim that predictions you see in the app today are calibrated — they are not until the artifact is wired live.</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function CalibrationEvidencePage() {
  return (
    <RequireAuth>
      <CalibrationEvidenceInner />
    </RequireAuth>
  );
}
