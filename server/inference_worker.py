import base64
import io
import json
from pathlib import Path
from typing import List, Optional

import torch
import time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from pydantic import BaseModel
from PIL import Image
import groundingdino.datasets.transforms as T
from torchvision.ops import nms
from starlette.middleware.base import BaseHTTPMiddleware

from demo.inference_on_a_image import load_model, get_grounding_output

CONFIG_PATH = Path("groundingdino/config/GroundingDINO_SwinT_OGC.py")
CHECKPOINT_PATH = Path("weights/groundingdino_swint_ogc.pth")

# ---------- Response models ----------
class InferenceParams(BaseModel):
    box_threshold: float
    text_threshold: Optional[float]
    nms_threshold: float
    max_detections: int
    return_visualization: bool

class DetectionResult(BaseModel):
    image_id: str
    boxes: List[List[float]]
    phrases: List[str]
    scores: List[float]
    visualization_b64: Optional[str] = None

class Metrics(BaseModel):
    forward_ms: float
    server_e2e_ms: float
    t_server_request_received: float
    t_server_response_done: float


class DetectResponse(BaseModel):
    request_id: str
    device: str
    results: List[DetectionResult]
    # total forward pass time in milliseconds across all images (server-side)
    forward_ms: Optional[float] = None
    metrics: Optional[Metrics] = None

# ---------- Preprocessing & helpers ----------
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def prepare_image(data: bytes):
    image_pil = Image.open(io.BytesIO(data)).convert("RGB")
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor

def tensor_boxes_to_xyxy(boxes, size):
    # cxcywh -> xyxy in pixel space
    W, H = size
    boxes = boxes * torch.tensor([W, H, W, H])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# ---------- Model manager ----------
class ModelManager:
    def __init__(self):
        self.models = {}

    def get_model(self, use_gpu: bool):
        wants_gpu = use_gpu and torch.cuda.is_available()
        device = "cuda" if wants_gpu else "cpu"
        if device not in self.models:
            self.models[device] = load_model(
                str(CONFIG_PATH),
                str(CHECKPOINT_PATH),
                cpu_only=(device == "cpu"),
            )
        return self.models[device], device

model_manager = ModelManager()
app = FastAPI(title="GroundingDINO Inference Service")


class E2ETimerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Server request received (approx) in epoch ms
        t0_ms = time.time() * 1000.0
        request.state.t_server_request_received = t0_ms
        response = await call_next(request)
        # Do not modify headers/body here; endpoint will compute metrics and include in body.
        return response


app.add_middleware(E2ETimerMiddleware)

# ---------- Main endpoint ----------
def _parse_optional_json_list(name: str, raw: Optional[str]) -> Optional[List]:
    """Parse optional JSON list from a string; treat empty/whitespace/null as None."""
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.lower() in ("null", "none"):
        return None
    try:
        val = json.loads(s)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse {name}: {exc}") from exc
    if not isinstance(val, list):
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON list")
    if len(val) == 0:
        # treat empty list as not provided
        return None
    return val


@app.post("/v1/detect", response_model=DetectResponse)
async def detect(
    request: Request,
    request_id: str = Form(...),
    use_gpu: bool = Form(True),
    prompt_text: str = Form(..., description="Text prompt; it's recommended to end with a period."),
    prompt_token_spans: Optional[str] = Form(
        None, description="JSON string like [[[0,4]],[[5,9]]]; optional"
    ),
    box_threshold: float = Form(0.35),
    text_threshold: Optional[float] = Form(0.25),
    nms_threshold: float = Form(0.5),
    max_detections: int = Form(100),
    return_visualization: bool = Form(False),
    image_ids: Optional[str] = Form(
        None, description="JSON list matching order of uploaded images; optional (defaults to filename)"
    ),
    images: List[UploadFile] = File(...),
):
    if not images:
        raise HTTPException(status_code=400, detail="images array cannot be empty")

    # robust parsing for optional JSON fields
    token_spans = _parse_optional_json_list("token_spans", prompt_token_spans)
    ids_list = _parse_optional_json_list("image_ids", image_ids)
    if ids_list is not None and len(ids_list) != len(images):
        raise HTTPException(
            status_code=400, detail="image_ids length must match number of uploaded images"
        )

    model, device = model_manager.get_model(use_gpu)

    params = InferenceParams(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
        return_visualization=return_visualization,
    )

    results = []
    forward_ms_total = 0.0

    for idx, upload in enumerate(images[: params.max_detections]):
        data = await upload.read()
        image_pil, image_tensor = prepare_image(data)
        image_id = ids_list[idx] if ids_list else (upload.filename or f"image_{idx}")

        # choose text_threshold depending on whether token_spans provided
        text_threshold_to_use = None if token_spans is not None else params.text_threshold

        # Forward timing (GPU requires synchronize for accurate measurement)
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd0 = time.perf_counter()

        boxes, phrases = get_grounding_output(
            model=model,
            image=image_tensor,
            caption=prompt_text,
            box_threshold=params.box_threshold,
            text_threshold=text_threshold_to_use,
            with_logits=True,
            cpu_only=(device == "cpu"),
            token_spans=token_spans,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()
        forward_ms_total += (t_fwd1 - t_fwd0) * 1000.0
        scores = [float(p[p.rfind("(")+1 : p.rfind(")")]) if "(" in p else 0.0 for p in phrases]
        phrases = [p.split("(")[0] if "(" in p else p for p in phrases]

        limit = min(len(boxes), params.max_detections)
        boxes = boxes[:limit]
        phrases = phrases[:limit]
        scores = scores[:limit]

        boxes_xyxy = tensor_boxes_to_xyxy(boxes.clone(), size=image_pil.size)
        if len(boxes_xyxy) > 0 and 0 <= params.nms_threshold < 1.0:
            score_tensor = torch.tensor(scores)
            keep = nms(boxes_xyxy, score_tensor, params.nms_threshold)
            boxes_xyxy = boxes_xyxy[keep]
            boxes = boxes[keep]
            phrases = [phrases[i] for i in keep.tolist()]
            scores = [scores[i] for i in keep.tolist()]

        result = DetectionResult(
            image_id=image_id,
            boxes=boxes_xyxy.tolist(),
            phrases=phrases,
            scores=scores,
        )
        if params.return_visualization:
            from demo.inference_on_a_image import plot_boxes_to_image
            vis_image, _ = plot_boxes_to_image(image_pil.copy(), {"size": image_pil.size[::-1], "boxes": boxes, "labels": phrases})
            buffered = io.BytesIO()
            vis_image.save(buffered, format="JPEG")
            result.visualization_b64 = base64.b64encode(buffered.getvalue()).decode()
        results.append(result)

    # Compose metrics in body using request.state timestamps from middleware
    t_server_request_received = getattr(request.state, "t_server_request_received", None)
    t1_ms = time.time() * 1000.0
    # Choose start time: middleware's t_server_request_received if present, else use now as best-effort
    start_ms = t_server_request_received if isinstance(t_server_request_received, (int, float)) else t1_ms
    server_e2e_ms = t1_ms - start_ms

    metrics = Metrics(
        forward_ms=forward_ms_total,
        server_e2e_ms=server_e2e_ms,
        t_server_request_received=float(start_ms),
        t_server_response_done=float(t1_ms),
    )

    return DetectResponse(request_id=request_id, device=device, results=results, forward_ms=forward_ms_total, metrics=metrics)

# 启动方式（示例）：
# uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
