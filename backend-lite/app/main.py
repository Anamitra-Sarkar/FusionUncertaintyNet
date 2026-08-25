"""Lite Backend — Render, Firebase Auth, Groq proxy, Firestore, forwards to Heavy."""
import os, time, json, base64
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

app = FastAPI(title="FusionUncertaintyNet Lite", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Firebase Admin lazy init ----
_firebase_inited = False
def init_firebase():
    global _firebase_inited
    if _firebase_inited:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials
        # try base64 env first
        b64 = os.getenv("FIREBASE_ADMIN_JSON_BASE64")
        if b64:
            try:
                decoded = base64.b64decode(b64).decode()
                cred = credentials.Certificate(json.loads(decoded))
                firebase_admin.initialize_app(cred)
                _firebase_inited = True
                print("[lite] Firebase initialized from base64")
                return
            except Exception as e:
                print(f"[lite] base64 firebase failed: {e}")
        # try file path
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("FIREBASE_ADMIN_JSON_PATH") or "./firebase-admin.json"
        if os.path.exists(path):
            cred = credentials.Certificate(path)
            firebase_admin.initialize_app(cred)
            _firebase_inited = True
            print(f"[lite] Firebase initialized from file {path}")
            return
        # fallback: try env json string
        j = os.getenv("FIREBASE_ADMIN_JSON")
        if j:
            cred = credentials.Certificate(json.loads(j))
            firebase_admin.initialize_app(cred)
            _firebase_inited = True
            print("[lite] Firebase initialized from json env")
            return
        print("[lite] No Firebase admin cred found — auth will be mocked for local dev")
    except Exception as e:
        print(f"[lite] Firebase init error: {e}")

init_firebase()

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    # if firebase not inited, allow but mark as mock user for local dev
    if not _firebase_inited:
        if token == "mock" or len(token) < 20:
            return {"uid": "mock-user", "email": "mock@test.com"}
        # try still verify if possible
        try:
            from firebase_admin import auth
            decoded = auth.verify_id_token(token)
            return decoded
        except Exception:
            # for local dev without firebase, accept any token as mock but log
            print("[lite] Firebase not inited, accepting token as mock")
            return {"uid": "mock-user", "email": "mock@test.com"}
    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# ---- Models ----
class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=5000)
    plddt: Optional[List[float]] = None
    phi: Optional[List[float]] = None
    psi: Optional[List[float]] = None
    pae: Optional[List[List[float]]] = None
    disorder_score: Optional[float] = Field(None, ge=0, le=1)

class ExplainRequest(BaseModel):
    sequence: str
    global_quality: float
    global_uncertainty: float
    gates: List[float]
    history_summary: Optional[str] = None

HEAVY_URL = os.getenv("HF_SPACE_URL", os.getenv("HEAVY_URL", "http://localhost:7860"))
HEAVY_SECRET = os.getenv("HEAVY_SHARED_SECRET", "change-me-32chars")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

@app.get("/health")
def health():
    return {"status": "ok", "heavy_url": HEAVY_URL, "firebase": _firebase_inited, "groq_model": GROQ_MODEL}

@app.get("/")
def root():
    return {"service": "FusionUncertaintyNet Lite", "docs": "/docs"}

@app.post("/api/predict")
async def predict_proxy(req: PredictRequest, user=Depends(verify_token)):
    """Verify auth, forward to heavy, log to Firestore, return."""
    heavy_endpoint = HEAVY_URL.rstrip("/") + "/predict"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                heavy_endpoint,
                json=req.model_dump(),
                headers={"X-Render-Secret": HEAVY_SECRET, "Content-Type": "application/json"}
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Heavy backend unreachable: {e}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Heavy error: {resp.text[:500]}")
        data = resp.json()

    # log to Firestore (namespaced fusion_predictions) — best effort, don't fail request if Firestore down
    # cabbage-guard Firestore database is named "default" (without parentheses) in asia-south1, not "(default)"
    try:
        if _firebase_inited:
            from firebase_admin import firestore
            # Try database_id="default" for firebase_admin>=6.6, fallback to database="default" or default
            try:
                db = firestore.client(database_id="default")
            except TypeError:
                try:
                    db = firestore.client(database="default")
                except TypeError:
                    db = firestore.client()
            job_id = f"{user['uid']}_{int(time.time()*1000)}"
            doc = {
                "uid": user["uid"],
                "email": user.get("email"),
                "sequence": data.get("sequence", req.sequence)[:500],  # truncate for privacy
                "length": data.get("length"),
                "global_quality": data.get("global_quality"),
                "global_uncertainty": data.get("global_uncertainty"),
                "gates": data.get("gates"),
                "ramachandran_outliers": data.get("ramachandran_outliers"),
                "created_at": firestore.SERVER_TIMESTAMP,
                "fusion_version": "0.1.0"
            }
            db.collection("fusion_predictions").document(job_id).set(doc)
            data["job_id"] = job_id
        else:
            data["job_id"] = f"mock_{int(time.time())}"
    except Exception as e:
        print(f"[lite] Firestore log failed: {e}")
        data["job_id"] = f"fallback_{int(time.time())}"

    return data

@app.get("/api/history")
def history(user=Depends(verify_token), limit: int = 20):
    if not _firebase_inited:
        return {"items": [], "note": "Firebase not configured — mock history"}
    try:
        from firebase_admin import firestore
        try:
            db = firestore.client(database_id="default")
        except TypeError:
            try:
                db = firestore.client(database="default")
            except TypeError:
                db = firestore.client()
        # query fusion_predictions where uid == user uid, order by created_at desc
        # Handle case where Firestore database not exists (cabbage-guard may need manual enable via console)
        try:
            q = db.collection("fusion_predictions").where("uid", "==", user["uid"]).order_by("created_at", direction=firestore.Query.DESCENDING).limit(min(limit, 50))
            docs = q.stream()
            items = []
            for d in docs:
                data = d.to_dict()
                data["id"] = d.id
                if "created_at" in data and hasattr(data["created_at"], "isoformat"):
                    data["created_at"] = data["created_at"].isoformat()
                items.append(data)
            return {"items": items}
        except Exception as inner:
            # Firestore not enabled -> return empty with note, do not crash
            if "does not exist" in str(inner) or "NOT_FOUND" in str(inner) or "404" in str(inner):
                print(f"[lite] Firestore not enabled, returning empty history: {inner}")
                return {"items": [], "note": "Firestore database not yet enabled for cabbage-guard — enable via https://console.cloud.google.com/datastore/setup?project=cabbage-guard or use mock history. Predictions still work and are returned directly."}
            raise
    except Exception as e:
        print(f"[lite] history error: {e}")
        # Graceful fallback instead of 500
        return {"items": [], "note": f"History temporarily unavailable: {str(e)[:200]}", "error": str(e)[:500]}

@app.post("/api/explain")
def explain(req: ExplainRequest, user=Depends(verify_token)):
    """Groq proxy — never exposes GROQ_API_KEY to client. Uses openai/gpt-oss-20b available for this key."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="Groq not configured")
    # rate limit simple: check sequence length
    if len(req.sequence) > 2000:
        req.sequence = req.sequence[:2000] + "...[truncated]"

    # craft prompt
    gates_desc = f"ESM-2 weight {req.gates[0]:.2f}, ProtT5 {req.gates[1]:.2f}, AF-features {req.gates[2]:.2f}" if len(req.gates)==3 else str(req.gates)
    prompt = f"""You are a structural biology assistant. Explain protein structure reliability.
Sequence (first 200 aa): {req.sequence[:200]}
Global predicted quality: {req.global_quality:.1f}/100
Global uncertainty: {req.global_uncertainty:.2f}
Adaptive gating: {gates_desc}
Task: Provide concise, factual assessment (3 bullet points + 1 paragraph) covering:
- What predicted quality means vs pLDDT
- Which gating source dominated and why (length/disorder context)
- Aleatoric vs epistemic uncertainty implications for wet-lab design
- Be calibrated, no hype, mention limitations."""
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a calibrated structural biology explainer. Be concise, quantitative, no emojis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        text = chat.choices[0].message.content
        # log choice if Groq returns reasoning field (some models)
        return {"explanation": text, "model": GROQ_MODEL}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Groq error: {e}")


class _PDBCache:
    ts: float = 0.0
    url: str = ""
_pdb_cache = _PDBCache()

@app.get("/api/pdb")
def get_afdb_pdb(acc: str):
    """Resolve latest AlphaFold model PDB for a UniProt accession (real AFDB)."""
    import re as _re, time as _time
    if not _re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{4}[0-9A-Z]|[A-NR-Z][0-9][A-Z][A-Z0-9]{3}[0-9]", acc or ""):
        raise HTTPException(status_code=400, detail="Invalid UniProt accession")
    now = _time.time()
    if not (_pdb_cache.url and now - _pdb_cache.ts < 86400 and _pdb_cache.url.endswith(acc)):
        try:
            r = httpx.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=15)
            if r.status_code != 200:
                raise HTTPException(status_code=404, detail="No AlphaFold model")
            meta = r.json()[0]
            ver = meta.get("latestVersion", 6)
            _pdb_cache.url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
            _pdb_cache.ts = now
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AFDB unreachable: {e}")
    try:
        pr = httpx.get(_pdb_cache.url, timeout=30)
        if pr.status_code != 200:
            raise HTTPException(status_code=404, detail="Model file missing")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")
    from fastapi import Response
    return Response(content=pr.text, media_type="text/plain",
                    headers={"Access-Control-Allow-Origin": "*",
                             "Cache-Control": "public, max-age=86400"})


@app.get("/api/me")
def me(user=Depends(verify_token)):
    return {"uid": user.get("uid"), "email": user.get("email")}

# ---- Firestore security note ----
# Collections are namespaced `fusion_*` to avoid colliding with other 35 projects on same cabbage-guard.
# Rules should be: match /fusion_predictions/{id} { allow read, write: if request.auth.uid == resource.data.uid; }
