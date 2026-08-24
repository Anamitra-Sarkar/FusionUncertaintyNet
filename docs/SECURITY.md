# Security & Privacy

- Login mandatory via Firebase Auth (cabbage-guard). `middleware` + backend `verify_id_token` on every call.
- Secrets via env only, never committed. `.gitignore` covers all.
- Collections `fusion_*` namespaced — does not touch other projects on same Firebase.
- Groq key proxied via lite backend only, rate limited, sequence truncated to 500 chars for LLM.
- Firestore TTL planned: auto-delete predictions after 30 days unless user pins.
- Heavy backend stateless, no PII, shared secret between lite/heavy.
- HF Space public but inference requires no auth for demo; production should gate via lite.
