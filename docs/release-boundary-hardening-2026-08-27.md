# Release-Boundary Hardening — 27 August 2026

FusionUncertaintyNet’s heavy service previously attempted Hub retrieval and then constructed a random-initialized model if a checkpoint could not be obtained. Its internal shared-secret check could also permit a request when the credential was absent or mismatched. Those behaviors could create a numerical output without an approved, reproducible model release.

The service now accepts inference only after all of the following conditions are met: `MODEL_RELEASE_APPROVED=true`; an immutable `MODEL_ARTIFACT_REVISION` is configured; a local checkpoint file is present at the configured `MODEL_PATH`; and a non-placeholder `HEAVY_SHARED_SECRET` authenticates the internal request. Otherwise `/ready` and prediction routes return explicit `503` status codes, and no random model is constructed. The public health response distinguishes liveness from whether a release is configured or loaded.

The lightweight gateway now requires Firebase Admin verification and a configured internal heavy-service credential. It does not mint mock identities, return mock history, expose raw authentication or persistence errors, or return an inference response when immutable audit persistence is unavailable. Both browser-facing and heavy-service CORS policies now require configured origins rather than permitting every origin with credentials.

| Local verification | Result |
|---|---|
| Backend release-boundary tests | 3 passed. |
| Heavy-service no-model readiness | Verified `503` with `MODEL_NOT_READY`. |
| Heavy-service internal authentication | Verified `503` when unconfigured and no model construction. |
| Lightweight Firebase boundary | Verified that `Bearer mock` returns `503` when Firebase verification is unavailable. |
| Frontend | Next.js `16.3.3` production build and direct ESLint run passed. |
| Dependency audit | Production audit passed with zero known vulnerabilities. |

This change intentionally makes an unconfigured service less permissive. It does not train, upload, promote, or make any biomedical or structural prediction claim.

The public landing page was also changed to remove its hard-coded “live example” sequence, quality score, uncertainty value, and gating weights. It now presents the same no-approved-artifact abstention state as the service and is protected by a source-level boundary regression. This prevents a promotional interface from implying a live, calibrated result where the independently reviewed release evidence is absent.

Shared metadata, navigation, and footer language now describe a release-gated research workspace rather than available calibrated scores or an operational prediction service. The public-boundary regression scans those shared surfaces as well as the landing page.

## Provider verification

The heavy Hugging Face Space was found publicly exposed during verification and was immediately restored to private visibility. Its source synchronization workflow completed successfully for the fail-closed hardening revision. Because the initially running container retained an older revision, the Space was restarted and then verified to report the synchronized source revision. Authenticated non-prediction checks returned health with `model_loaded: false`, `release_configured: false`, and `/ready` with HTTP `503` and `MODEL_NOT_READY`.

The public Vercel frontend deployment `dpl_7KNNFbG8RLzwAraanbcmEKCWoEVo` for `04f8878` reached `READY`. The canonical landing page was checked and presented the no-approved-artifact abstention state.

The lightweight Render service deployed the hardening revision `04f8878`. Its health response reports that identity verification and heavy internal authentication are configured without exposing their values. A browser preflight from `https://protein-reliability.vercel.app` returned HTTP `200` with that exact `Access-Control-Allow-Origin` value; it does not return a wildcard or credentialed cross-origin response.

## CI compatibility verification

The frontend’s controlled Next.js `16.3.3` update requires Node.js `20.9.0` or newer. The repository CI workflow previously selected Node.js 18, causing the frontend build to fail before it could exercise the release-boundary source. Updating only the runner baseline to Node.js 20 restored the completed CI run `33120702982` for `9854081`. The same source built locally with explicit production environment settings, generated all six static routes, and the three heavy/lite fail-closed boundary regressions passed. The heavy Space synchronization workflow remains path-filtered to `backend-heavy/**`, `training/**`, and its own workflow definition; this CI correction did not sync, release, or load a model artifact.
