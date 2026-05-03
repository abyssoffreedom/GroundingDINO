import base64
import asyncio
import io
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from pydantic import BaseModel
from PIL import Image, ImageOps
# Enable HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow_heif not installed. HEIC images will not be supported.")

import groundingdino.datasets.transforms as T
from torchvision.ops import nms
from demo.inference_on_a_image import load_model, get_grounding_output, plot_boxes_to_image
from server.monotonic_clock import server_now_ns
from server.udp_echo_server import start_udp_echo_server

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
    boxes_coco: Optional[List[List[float]]] = None
    phrases: List[str]
    scores: List[float]
    visualization_b64: Optional[str] = None

class Metrics(BaseModel):
    server_processing_ms: float
    server_post_body_parse_ms: float
    server_request_parse_ms: float
    server_model_lookup_ms: float
    server_upload_read_ms: float
    server_image_decode_ms: float
    server_image_transform_ms: float
    server_inference_ms: float
    server_postprocess_ms: float
    server_visualization_ms: float
    server_receive_ns: int
    server_send_ns: int
    server_request_start_ns: int
    server_request_body_complete_ns: int
    server_response_ready_ns: int
    request_body_bytes: int
    uploaded_image_bytes: int
    uploaded_image_width: Optional[int] = None
    uploaded_image_height: Optional[int] = None
    model_tensor_width: Optional[int] = None
    model_tensor_height: Optional[int] = None
    boxes_before_nms: Optional[int] = None
    boxes_after_nms: Optional[int] = None


class DetectResponse(BaseModel):
    request_id: str
    device: str
    results: List[DetectionResult]
    metrics: Optional[Metrics] = None

# ---------- Preprocessing & helpers ----------
transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def elapsed_ms(start_ns: int, end_ns: int) -> float:
    return max(0, end_ns - start_ns) / 1_000_000.0


def prepare_image(data: bytes):
    decode_start_ns = server_now_ns()
    image_pil = Image.open(io.BytesIO(data))
    # 自动根据 EXIF 信息旋转图片（解决手机照片侧躺问题）
    image_pil = ImageOps.exif_transpose(image_pil)
    image_pil = image_pil.convert("RGB")
    decode_end_ns = server_now_ns()
    transform_start_ns = decode_end_ns
    image_tensor, _ = transform(image_pil, None)
    transform_end_ns = server_now_ns()
    return (
        image_pil,
        image_tensor,
        elapsed_ms(decode_start_ns, decode_end_ns),
        elapsed_ms(transform_start_ns, transform_end_ns),
    )

def tensor_boxes_to_xyxy(boxes, size):
    # cxcywh -> xyxy in pixel space
    W, H = size
    boxes = boxes.clone() * torch.tensor([W, H, W, H])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def tensor_boxes_to_coco_xywh(boxes, size):
    # cxcywh normalized -> [xmin, ymin, w, h] in pixel space
    W, H = size
    boxes = boxes.clone() * torch.tensor([W, H, W, H])
    boxes[:, :2] -= boxes[:, 2:] / 2 # Convert cx, cy to xmin, ymin
    return boxes

def tensor_boxes_to_normalized_xyxy(boxes):
    # cxcywh -> xyxy in normalized [0, 1] space
    boxes = boxes.clone()
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
UDP_ECHO_HOST = "0.0.0.0"
UDP_ECHO_PORT = 9999
udp_echo_transport = None
udp_echo_process = None


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _winsock_udp_server_executable() -> Path:
    return Path(__file__).with_name("winsock_timestamp_udp_server.exe")


def _winsock_udp_server_source() -> Path:
    return Path(__file__).with_name("winsock_timestamp_udp_server.cpp")


def _make_winsock_udp_server_command() -> Optional[List[str]]:
    if os.name != "nt":
        return None

    if not _env_flag_enabled("NETWORK_PROBE_USE_WINSOCK_TIMESTAMP", default=True):
        return None

    executable = _winsock_udp_server_executable()
    if not executable.exists():
        return None

    source = _winsock_udp_server_source()
    if source.exists() and executable.stat().st_mtime < source.stat().st_mtime:
        print(
            "[NetworkProbe] Windows native UDP probe server is older than "
            "winsock_timestamp_udp_server.cpp; rebuild it before using native PTR timestamps. "
            "Falling back to Python UDP server."
        )
        return None

    command = [
        str(executable),
        "--host",
        UDP_ECHO_HOST,
        "--port",
        str(UDP_ECHO_PORT),
    ]

    high_priority = os.environ.get("NETWORK_PROBE_HIGH_PRIORITY", "1")
    command.extend(["--high-priority", high_priority])

    return command


class HTTPTransferTimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        server_request_start_ns = server_now_ns()
        state["server_request_start_ns"] = server_request_start_ns
        # Backward-compatible alias used by older clients.
        state["server_receive_ns"] = server_request_start_ns
        state["request_body_bytes"] = 0

        async def timing_receive():
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                state["request_body_bytes"] = int(state.get("request_body_bytes", 0)) + len(body)
                if not message.get("more_body", False):
                    state.setdefault("server_request_body_complete_ns", server_now_ns())
            return message

        async def timing_send(message):
            if message["type"] == "http.response.start":
                server_response_send_start_ns = server_now_ns()
                state["server_response_send_start_ns"] = server_response_send_start_ns
                headers = list(message.get("headers", []))
                headers.append((
                    b"x-server-response-send-start-ns",
                    str(server_response_send_start_ns).encode("ascii"),
                ))
                if "server_request_body_complete_ns" in state:
                    headers.append((
                        b"x-server-request-body-complete-ns",
                        str(state["server_request_body_complete_ns"]).encode("ascii"),
                    ))
                message["headers"] = headers
            await send(message)

        await self.app(scope, timing_receive, timing_send)


app.add_middleware(HTTPTransferTimingMiddleware)


@app.on_event("startup")
async def startup_event():
    global udp_echo_transport, udp_echo_process
    winsock_command = _make_winsock_udp_server_command()
    if winsock_command is not None:
        udp_echo_process = subprocess.Popen(
            winsock_command,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        await asyncio.sleep(0.2)
        if udp_echo_process.poll() is None:
            print(
                "[NetworkProbe] started Windows native UDP probe server: "
                + " ".join(winsock_command)
            )
            return

        print(
            "[NetworkProbe] Windows native UDP probe server exited during startup; "
            "falling back to Python UDP server."
        )
        udp_echo_process = None

    udp_echo_transport = await start_udp_echo_server(
        host=UDP_ECHO_HOST,
        port=UDP_ECHO_PORT,
    )


@app.on_event("shutdown")
async def shutdown_event():
    global udp_echo_transport, udp_echo_process
    if udp_echo_process is not None:
        udp_echo_process.terminate()
        try:
            udp_echo_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            udp_echo_process.kill()
            udp_echo_process.wait(timeout=2)
        udp_echo_process = None

    if udp_echo_transport is not None:
        udp_echo_transport.close()
        udp_echo_transport = None

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
    t_process_start_ns = server_now_ns()
    server_request_body_complete_ns = getattr(request.state, "server_request_body_complete_ns", None)
    if server_request_body_complete_ns is None:
        server_request_body_complete_ns = t_process_start_ns
    server_post_body_parse_ms = elapsed_ms(
        int(server_request_body_complete_ns),
        t_process_start_ns,
    )

    if not images:
        raise HTTPException(status_code=400, detail="images array cannot be empty")

    # robust parsing for optional JSON fields
    request_parse_start_ns = server_now_ns()
    token_spans = _parse_optional_json_list("token_spans", prompt_token_spans)
    ids_list = _parse_optional_json_list("image_ids", image_ids)
    if ids_list is not None and len(ids_list) != len(images):
        raise HTTPException(
            status_code=400, detail="image_ids length must match number of uploaded images"
        )
    request_parse_end_ns = server_now_ns()

    model_lookup_start_ns = server_now_ns()
    model, device = model_manager.get_model(use_gpu)
    model_lookup_end_ns = server_now_ns()

    params = InferenceParams(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
        return_visualization=return_visualization,
    )

    results = []
    uploaded_image_bytes = 0
    server_upload_read_ms = 0.0
    server_image_decode_ms = 0.0
    server_image_transform_ms = 0.0
    server_inference_ms = 0.0
    server_postprocess_ms = 0.0
    server_visualization_ms = 0.0
    uploaded_image_width = None
    uploaded_image_height = None
    model_tensor_width = None
    model_tensor_height = None
    boxes_before_nms = 0
    boxes_after_nms = 0

    for idx, upload in enumerate(images[: params.max_detections]):
        upload_read_start_ns = server_now_ns()
        data = await upload.read()
        upload_read_end_ns = server_now_ns()
        server_upload_read_ms += elapsed_ms(upload_read_start_ns, upload_read_end_ns)
        uploaded_image_bytes += len(data)
        image_pil, image_tensor, image_decode_ms, image_transform_ms = prepare_image(data)
        server_image_decode_ms += image_decode_ms
        server_image_transform_ms += image_transform_ms
        uploaded_image_width = image_pil.size[0]
        uploaded_image_height = image_pil.size[1]
        if hasattr(image_tensor, "shape") and len(image_tensor.shape) >= 3:
            model_tensor_height = int(image_tensor.shape[-2])
            model_tensor_width = int(image_tensor.shape[-1])
        image_id = ids_list[idx] if ids_list else (upload.filename or f"image_{idx}")

        # choose text_threshold depending on whether token_spans provided
        text_threshold_to_use = None if token_spans is not None else params.text_threshold

        if device == "cuda":
            torch.cuda.synchronize()
        inference_start_ns = server_now_ns()
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
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
        inference_end_ns = server_now_ns()
        server_inference_ms += elapsed_ms(inference_start_ns, inference_end_ns)

        postprocess_start_ns = server_now_ns()
        boxes_before_nms += len(boxes)
        scores = [float(p[p.rfind("(")+1 : p.rfind(")")]) if "(" in p else 0.0 for p in phrases]
        phrases = [p.split("(")[0] if "(" in p else p for p in phrases]

        limit = min(len(boxes), params.max_detections)
        boxes = boxes[:limit]
        phrases = phrases[:limit]
        scores = scores[:limit]

        # Return normalized xyxy coordinates [0, 1]
        boxes_xyxy = tensor_boxes_to_normalized_xyxy(boxes)
        # Return COCO format [xmin, ymin, w, h] in pixels
        boxes_coco = tensor_boxes_to_coco_xywh(boxes, size=image_pil.size)

        if len(boxes_xyxy) > 0 and 0 <= params.nms_threshold < 1.0:
            score_tensor = torch.tensor(scores)
            keep = nms(boxes_xyxy, score_tensor, params.nms_threshold)
            boxes_xyxy = boxes_xyxy[keep]
            boxes_coco = boxes_coco[keep]
            boxes = boxes[keep]
            phrases = [phrases[i] for i in keep.tolist()]
            scores = [scores[i] for i in keep.tolist()]
        boxes_after_nms += len(boxes_xyxy)

        result = DetectionResult(
            image_id=image_id,
            boxes=boxes_xyxy.tolist(),
            boxes_coco=boxes_coco.tolist(),
            phrases=phrases,
            scores=scores,
        )
        postprocess_end_ns = server_now_ns()
        server_postprocess_ms += elapsed_ms(postprocess_start_ns, postprocess_end_ns)

        if params.return_visualization:
            visualization_start_ns = server_now_ns()
            vis_image, _ = plot_boxes_to_image(image_pil.copy(), {"size": image_pil.size[::-1], "boxes": boxes, "labels": phrases})
            buffered = io.BytesIO()
            vis_image.save(buffered, format="JPEG")
            result.visualization_b64 = base64.b64encode(buffered.getvalue()).decode()
            visualization_end_ns = server_now_ns()
            server_visualization_ms += elapsed_ms(visualization_start_ns, visualization_end_ns)
        results.append(result)

    # Compose metrics using the same monotonic clock as UDP time-sync probes.
    server_request_start_ns = getattr(request.state, "server_request_start_ns", None)
    if server_request_start_ns is None:
        server_request_start_ns = getattr(request.state, "server_receive_ns", t_process_start_ns)

    request_body_bytes = int(getattr(request.state, "request_body_bytes", 0) or 0)
    server_response_ready_ns = server_now_ns()
    server_processing_ms = (
        server_response_ready_ns - int(server_request_body_complete_ns)
    ) / 1_000_000.0

    metrics = Metrics(
        server_processing_ms=server_processing_ms,
        server_post_body_parse_ms=server_post_body_parse_ms,
        server_request_parse_ms=elapsed_ms(request_parse_start_ns, request_parse_end_ns),
        server_model_lookup_ms=elapsed_ms(model_lookup_start_ns, model_lookup_end_ns),
        server_upload_read_ms=server_upload_read_ms,
        server_image_decode_ms=server_image_decode_ms,
        server_image_transform_ms=server_image_transform_ms,
        server_inference_ms=server_inference_ms,
        server_postprocess_ms=server_postprocess_ms,
        server_visualization_ms=server_visualization_ms,
        server_receive_ns=int(server_request_start_ns),
        server_send_ns=int(server_response_ready_ns),
        server_request_start_ns=int(server_request_start_ns),
        server_request_body_complete_ns=int(server_request_body_complete_ns),
        server_response_ready_ns=int(server_response_ready_ns),
        request_body_bytes=request_body_bytes,
        uploaded_image_bytes=uploaded_image_bytes,
        uploaded_image_width=uploaded_image_width,
        uploaded_image_height=uploaded_image_height,
        model_tensor_width=model_tensor_width,
        model_tensor_height=model_tensor_height,
        boxes_before_nms=boxes_before_nms,
        boxes_after_nms=boxes_after_nms,
    )

    return DetectResponse(request_id=request_id, device=device, results=results, metrics=metrics)

# 启动方式（示例）：
# uvicorn server.inference_worker:app --host 0.0.0.0 --port 8000
