# Project TODO

- [x] Add committed non-interactive ESLint configuration so the existing frontend lint command does not block release verification with a setup prompt.
- [x] Review and safely remediate the frontend dependency-audit findings through controlled, tested upgrades; do not use forced automated upgrades.
- [x] Run the full frontend/backend quality gate and reverify the existing Vercel, lightweight gateway, and heavy-service health boundaries without changing model or calibration claims.
- [x] Configure the lightweight gateway’s restrictive CORS allowlist for the canonical Vercel origin and verify the authenticated browser preflight succeeds without reintroducing wildcard access.
- [x] Add a focused backend health-contract regression that proves an unloaded heavy model remains visibly not ready and does not expose a prediction claim.
- [x] Remove random-initialized inference and fail closed when an approved checkpoint is unavailable; require a configured internal credential for heavy prediction routes and add regressions for both denial paths.
- [x] Remove the public hard-coded “live example” quality, uncertainty, and gating values, replacing them with an explicit no-approved-model state that does not represent a biomedical or structural prediction.
- [x] Remove shared footer and navigation language that implies calibrated scores or an available prediction workflow while no approved model artifact is configured.
