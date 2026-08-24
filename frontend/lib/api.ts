import { getIdToken } from "./firebase";

const LITE = process.env.NEXT_PUBLIC_LITE_URL || "http://localhost:8000";

async function authedFetch(path: string, opts: RequestInit = {}) {
  const token = await getIdToken();
  const res = await fetch(`${LITE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(opts.headers || {}),
    },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status}: ${txt.slice(0, 500)}`);
  }
  return res.json();
}

export const predict = (payload: any) => authedFetch("/api/predict", { method: "POST", body: JSON.stringify(payload) });
export const history = () => authedFetch("/api/history");
export const explain = (payload: any) => authedFetch("/api/explain", { method: "POST", body: JSON.stringify(payload) });
export const me = () => authedFetch("/api/me");
