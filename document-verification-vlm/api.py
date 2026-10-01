"""FastAPI service.  Run: uvicorn api:app --reload"""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from verifier.pipeline import verify_document
from verifier.vlm import PROMPTS, sniff_media_type

app = FastAPI(title="Document Verification API", version="1.0")
MAX_BYTES = 5 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok", "prompt_versions": list(PROMPTS)}


@app.post("/verify")
async def verify(file: UploadFile = File(...), prompt_version: str = Query("v2_schema_strict")):
    """Upload an ID image (PNG/JPEG, <=5 MB). Image is processed in memory and not stored;
    sensitive fields in the response are masked."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (5 MB max)")
    try:
        sniff_media_type(data)  # trust magic bytes, not the client's content-type
        return verify_document(data, prompt_version)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception:
        raise HTTPException(502, "Upstream model error; please retry")
