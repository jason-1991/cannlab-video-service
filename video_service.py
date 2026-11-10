#!/usr/bin/env python3
"""Minimal Ascend NPU video generation service (Wan2.1 family, torch_npu).

Async job contract:
  GET  /health                    -> {"status":"ok","model_loaded":bool,...}
  POST /generate-video            -> {"job_id": "...", "status":"queued"}
       JSON body: {"prompt": str,
                   "image_url": str?, "image_b64": str?,
                   "width": int, "height": int, "seconds": float,
                   "seed": int?, "steps": int?, "fps": int?}
       header X-Service-Token: <token>
  GET  /jobs/{job_id}             -> {"status": "queued|running|done|failed",
                                     "error": str?, "elapsed_s": float?}
  GET  /jobs/{job_id}/result      -> video/mp4 bytes (H.264) when done

Model selection by MODE env: "t2v" (Wan2.1-T2V-1.3B) or "i2v" (Wan2.1-I2V-1.3B,
both-weights build that also does image conditioning). Token read from
$SERVICE_TOKEN_FILE. Activity timestamp maintained for the idle watchdog.
"""
import base64
import io
import json
import os
import threading
import time
import traceback
import uuid

import requests
import torch
import torch_npu  # noqa: F401
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from PIL import Image

BASE = os.path.expanduser("~/canvas-service")
TOKEN_FILE = os.environ.get("SERVICE_TOKEN_FILE", os.path.join(BASE, "token.txt"))
ACTIVITY_FILE = os.environ.get("ACTIVITY_FILE", os.path.join(BASE, "last_activity"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE, "models"))
MODE = os.environ.get("MODE", "t2v")  # t2v | i2v
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    TOKEN = f.read().strip()


def touch_activity() -> None:
    try:
        with open(ACTIVITY_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(time.time())))
    except OSError:
        pass


app = FastAPI(title="canvas-npu-video")
pipe = None
pipe_lock = threading.Lock()
jobs = {}
jobs_lock = threading.Lock()


def model_dir() -> str:
    name = "wan2.1-i2v-1.3b-diffusers" if MODE == "i2v" else "wan2.1-t2v-1.3b-diffusers"
    return os.path.join(MODELS_DIR, name)


def load_pipe():
    global pipe
    if pipe is not None:
        return pipe
    from diffusers import WanImageToVideoPipeline, WanPipeline

    # Diffusers-layout weights: Wan-AI/Wan2.1-T2V-1.3B-Diffusers (and, if published,
    # an I2V 1.3B diffusers build). The original-layout weights under
    # models/Wan2.1-* are for the official generate.py CLI proof run only.
    if MODE == "i2v":
        p = WanImageToVideoPipeline.from_pretrained(
            model_dir(), torch_dtype=torch.float16
        )
    else:
        p = WanPipeline.from_pretrained(model_dir(), torch_dtype=torch.float16)
    p = p.to("npu")
    pipe = p
    return pipe


@app.on_event("startup")
def _startup() -> None:
    touch_activity()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": MODE,
        "model_loaded": pipe is not None,
        "device": "npu" if torch.npu.is_available() else "unavailable",
        "model_dir": model_dir(),
    }


def _load_first_frame(body: dict):
    if body.get("image_b64"):
        raw = base64.b64decode(body["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if body.get("image_url"):
        r = requests.get(body["image_url"], timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


def _run_job(job_id: str, body: dict) -> None:
    started = time.time()
    with jobs_lock:
        jobs[job_id]["status"] = "running"
    touch_activity()
    try:
        width = int(body.get("width", 832))
        height = int(body.get("height", 480))
        seconds = float(body.get("seconds", 5))
        fps = int(body.get("fps", 16))
        steps = int(body.get("steps", 30))
        seed = int(body.get("seed", 42))
        prompt = str(body.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt required")
        num_frames = max(5, int(round(seconds * fps)) )
        # Wan expects 4n+1 frames
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        img = _load_first_frame(body)
        with pipe_lock:
            p = load_pipe()
            gen = torch.Generator(device="npu").manual_seed(seed)
            kwargs = dict(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                generator=gen,
            )
            if img is not None:
                if MODE != "i2v":
                    raise ValueError("image input requires MODE=i2v")
                img = img.resize((width, height))
                kwargs["image"] = img
            frames = p(**kwargs).frames[0]
        out_path = os.path.join(OUT_DIR, f"{job_id}.mp4")
        _export_mp4(frames, out_path, fps)
        with jobs_lock:
            jobs[job_id].update(
                status="done", result_path=out_path,
                elapsed_s=round(time.time() - started, 1),
            )
    except Exception as e:  # noqa: BLE001
        with jobs_lock:
            jobs[job_id].update(
                status="failed",
                error=f"{type(e).__name__}: {e}",
                trace=traceback.format_exc()[-3000:],
                elapsed_s=round(time.time() - started, 1),
            )
    finally:
        touch_activity()


def _export_mp4(frames, out_path: str, fps: int) -> None:
    """Encode frames to H.264 MP4. Prefer imageio[ffmpeg]; fallback diffusers export."""
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
        )
        for fr in frames:
            writer.append_data(fr)
        writer.close()
        return
    except Exception:
        pass
    from diffusers.utils import export_to_video

    export_to_video(frames, out_path, fps=fps)


def _check_token(x_service_token: str) -> None:
    if x_service_token != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


@app.post("/generate-video")
async def generate_video(req: Request, x_service_token: str = Header(default="")):
    _check_token(x_service_token)
    body = await req.json()
    job_id = uuid.uuid4().hex[:16]
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "created": time.time(),
                        "request": {k: v for k, v in body.items() if k != "image_b64"}}
    touch_activity()
    threading.Thread(target=_run_job, args=(job_id, body), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_service_token: str = Header(default="")):
    _check_token(x_service_token)
    with jobs_lock:
        j = jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    return {k: v for k, v in j.items() if k not in ("result_path", "request")}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str, x_service_token: str = Header(default="")):
    _check_token(x_service_token)
    with jobs_lock:
        j = jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="no such job")
    if j["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job status: {j['status']}")
    with open(j["result_path"], "rb") as f:
        data = f.read()
    return Response(content=data, media_type="video/mp4")
