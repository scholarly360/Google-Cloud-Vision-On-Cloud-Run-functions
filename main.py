# main.py used for gcv on gcp - cloud run functions (gen2)
import io
import json
import os
import re
import uuid
from typing import Dict, List, Optional
from fastapi import Body
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from vellox import Vellox

from google.cloud import vision_v1 as vision
from google.cloud import storage
from google.protobuf.json_format import MessageToDict

# -------------------- Security (HTTP Bearer) --------------------
bearer_scheme = HTTPBearer(auto_error=False)
# Comma-separated tokens, e.g. "token1,token2"
_ALLOWED_TOKENS = {
    t.strip() for t in os.getenv("API_BEARER_TOKENS", "").split(",") if t.strip()
}

def require_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> str:
    """Require a Bearer token and validate it against _ALLOWED_TOKENS."""
    if credentials is None or not credentials.scheme or credentials.scheme.lower() != "bearer":
        # Not authenticated
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials or ""
    if not _ALLOWED_TOKENS:
        # Misconfiguration guard: no tokens configured
        raise HTTPException(
            status_code=500,
            detail="Server auth not configured: set API_BEARER_TOKENS env var",
        )
    if token not in _ALLOWED_TOKENS:
        # Authenticated but not authorized
        raise HTTPException(
            status_code=403,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
# ---------------------------------------------------------------

app = FastAPI(title="OCR API (Vision)", version="1.1")

@app.get("/health")
def health(_: str = Depends(require_bearer_token)):
    return {"status": "ok"}


# --- clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

# --- env
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET")          # bucket to upload source PDFs/TIFFs
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", UPLOAD_BUCKET)  # bucket to receive Vision JSON
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "gcv_vision_ocr")       # top-level prefix for results

# ---------- helpers ----------
def _require_buckets():
    if not UPLOAD_BUCKET:
        raise HTTPException(500, "UPLOAD_BUCKET env var not set")
    if not OUTPUT_BUCKET:
        raise HTTPException(500, "OUTPUT_BUCKET env var not set")

def _gcs_uri(bucket: str, path: str) -> str:
    path = path.lstrip("/")
    return f"gs://{bucket}/{path}"

def _parse_gcs_uri(uri: str) -> (str, str):
    m = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not m:
        raise HTTPException(400, f"Invalid GCS URI: {uri}")
    return m.group(1), m.group(2)

def _upload_bytes_to_gcs(data: bytes, bucket: str, dest_path: str) -> str:
    b = storage_client.bucket(bucket)
    blob = b.blob(dest_path)
    blob.upload_from_file(io.BytesIO(data), rewind=True)
    return _gcs_uri(bucket, dest_path)

def _start_async_vision(gcs_input_uri: str, gcs_output_prefix: str, batch_size: int = 20) -> str:
    # Build async request
    input_cfg = vision.InputConfig(
        gcs_source=vision.GcsSource(uri=gcs_input_uri),
        mime_type="application/pdf" if gcs_input_uri.lower().endswith(".pdf") else "image/tiff",
    )
    output_cfg = vision.OutputConfig(
        gcs_destination=vision.GcsDestination(uri=gcs_output_prefix),
        batch_size=batch_size,
    )
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    async_req = vision.AsyncAnnotateFileRequest(
        input_config=input_cfg,
        features=[feature],
        output_config=output_cfg,
    )

    op = vision_client.async_batch_annotate_files(requests=[async_req])
    # Return the fully-qualified operation name
    return op.operation.name

def _list_json_blobs(prefix_uri: str) -> List[str]:
    bucket, prefix = _parse_gcs_uri(prefix_uri)
    # Ensure trailing slash prefixes list under this "folder"
    if not prefix.endswith("/"):
        prefix += "/"
    blobs = storage_client.list_blobs(bucket, prefix=prefix)
    return [f"gs://{bucket}/{b.name}" for b in blobs if b.name.lower().endswith(".json")]

def _download_texts_from_outputs(prefix_uri: str) -> Dict:
    """Aggregate text across all Vision JSON shards into pages[]"""
    bucket_name, prefix = _parse_gcs_uri(prefix_uri)
    if not prefix.endswith("/"):
        prefix += "/"
    bucket = storage_client.bucket(bucket_name)

    pages = []
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        if not blob.name.lower().endswith(".json"):
            continue
        data = bucket.blob(blob.name).download_as_text()
        obj = json.loads(data)
        # Each file holds responses for N pages (based on batch_size).
        for file_resp in obj.get("responses", []):
            # Each file_resp is an AnnotateImageResponse
            full = file_resp.get("fullTextAnnotation", {})
            text = full.get("text", "")
            pages.append({"text": text, "source": f"gs://{bucket_name}/{blob.name}"})
    # Stabilize order by blob name then index
    pages_sorted = sorted(pages, key=lambda p: p["source"])
    # add page numbers
    for i, p in enumerate(pages_sorted, start=1):
        p["page"] = i
    return {"pages": pages_sorted, "file_count": len(_list_json_blobs(prefix_uri))}

# ---------- existing sync endpoint (images + small PDFs) ----------
from pypdf import PdfReader
@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    mode: str = Query("auto"),
    _: str = Depends(require_bearer_token),
):
    """
    Images => document_text_detection
    PDFs (<=5 pages) => online small-batch; longer PDFs raise 413 with guidance
    """
    ct = (file.content_type or "").lower()
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    if ct.startswith("image/"):
        image = vision.Image(content=data)
        resp = vision_client.document_text_detection(image=image)
        if resp.error.message:
            raise HTTPException(500, resp.error.message)
        return {"kind": "image", "pages": [{"page": 1, "text": resp.full_text_annotation.text or ""}]}

    if ct == "application/pdf":
        with io.BytesIO(data) as bio:
            bio.seek(0)
            try:
                num = len(PdfReader(bio).pages)
            except Exception:
                num = 9999
        if mode != "async" and num <= 5:
            # small batch online (sync)
            input_cfg = vision.InputConfig(mime_type="application/pdf", content=data)
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
            areq = vision.AnnotateFileRequest(input_config=input_cfg, features=features, pages=list(range(1, num+1)))
            response = vision_client.batch_annotate_files(requests=[areq])
            out_pages = []
            for i, r in enumerate(response.responses[0].responses, start=1):
                if r.error.message:
                    raise HTTPException(500, r.error.message)
                txt = r.full_text_annotation.text if r.full_text_annotation else ""
                out_pages.append({"page": i, "text": txt})
            return {"kind": "pdf", "pages": out_pages}
        raise HTTPException(413, f"PDF has {num} pages; use /ocr/async/start for large PDFs.")
    raise HTTPException(415, f"Unsupported content-type: {ct}")

# ---------- NEW: Async (GCS) endpoints ----------
@app.post("/ocr/async/start")
async def ocr_async_start(
    file: UploadFile = File(...),
    batch_size: int = Query(20, ge=1, le=100, description="Pages per JSON shard"),
    hint_prefix: Optional[str] = Query(None, description="Optional subfolder hint for outputs"),
    _: str = Depends(require_bearer_token),
):
    """
    Upload PDF/TIFF to GCS, start Vision async OCR job, return operation + GCS paths.
    """
    _require_buckets()
    ct = (file.content_type or "").lower()
    if ct not in ("application/pdf", "image/tiff", "image/tif"):
        raise HTTPException(415, "Only application/pdf or image/tiff allowed for async")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    job_id = str(uuid.uuid4())[:8]
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename or "document.pdf")
    src_path = f"uploads/{job_id}/{safe_name}"
    gcs_input = _upload_bytes_to_gcs(data, UPLOAD_BUCKET, src_path)

    sub = hint_prefix.strip("/") if hint_prefix else f"jobs/{job_id}"
    gcs_output_prefix = _gcs_uri(OUTPUT_BUCKET, f"{OUTPUT_ROOT}/{sub}/")

    op_name = _start_async_vision(gcs_input, gcs_output_prefix, batch_size=batch_size)
    return {
        "operation": op_name,
        "gcs_input": gcs_input,
        "gcs_output_prefix": gcs_output_prefix,
        "note": "Poll /ocr/async/status?operation=...; when done, GET /ocr/async/result?prefix=gs://bucket/path/",
    }

@app.get("/ocr/async/status")
def ocr_async_status(
    operation: str = Query(..., description="projects/.../operations/..."),
    _: str = Depends(require_bearer_token),
):
    """
    Poll Vision long-running operation.
    """
    # Use the transport's operations client to fetch the operation proto.
    op_pb = vision_client.transport.operations_client.get_operation(operation)
    # Convert to JSON-serializable dict
    info = MessageToDict(op_pb._pb) if hasattr(op_pb, "_pb") else MessageToDict(op_pb)
    return info

@app.get("/ocr/async/result")
def ocr_async_result(
    prefix: str = Query(..., description="gs://bucket/path/ to Vision output JSONs"),
    _: str = Depends(require_bearer_token),
):
    """
    Read all Vision JSON shards under the prefix and aggregate text.
    """
    return _download_texts_from_outputs(prefix)

# ---- Vellox entrypoint for Cloud Run functions
vellox = Vellox(app=app, lifespan="off")
def handler(request):
    return vellox(request)
