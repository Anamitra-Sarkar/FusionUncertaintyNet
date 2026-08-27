import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const page = readFileSync(resolve("app/page.tsx"), "utf8");
const prohibitedPublicExampleMarkers = [
  "LIVE EXAMPLE",
  "84.2",
  "ESM 0.42 · ProtT5 0.33 · AF 0.25",
  "501k AFdb training",
  "Run Prediction",
];

for (const marker of prohibitedPublicExampleMarkers) {
  if (page.includes(marker)) {
    throw new Error(`Public landing page must not include fabricated or release-unsupported marker: ${marker}`);
  }
}

for (const requiredStatement of ["MODEL RELEASE STATUS", "Inference is not available.", "No approved immutable model artifact"]) {
  if (!page.includes(requiredStatement)) {
    throw new Error(`Public landing page is missing required abstention statement: ${requiredStatement}`);
  }
}

console.log("Public research boundary regression passed.");
