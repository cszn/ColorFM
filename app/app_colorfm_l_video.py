from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Iterator

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    module=r"gradio\.routes",
)

try:
    import cv2
except ImportError:
    cv2 = None

import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.colorfm_l import ColorFM_L


DEFAULT_WEIGHTS_PATH = REPO_DIR / "checkpoints" / "colorfm_l.pth"
WEIGHTS_PATH = Path(os.getenv("COLORFM_L_WEIGHTS", DEFAULT_WEIGHTS_PATH)).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    os.getenv(
        "COLORFM_L_VIDEO_OUTPUT_DIR",
        REPO_DIR / "outputs" / "colorfm_l_video",
    )
).expanduser()
DEFAULT_FFMPEG_PATH = os.getenv("FFMPEG_BINARY", "")
KERNEL_IMAGE_SIZE = 256
UPLOAD_CACHE_CLEANUP_INTERVAL = 60 * 60
UPLOAD_CACHE_MAX_AGE = 24 * 60 * 60

DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]
DTYPE_CHOICES = ["auto", "bf16", "fp16", "fp32"]

RESOLUTION_LIMITS = {
    "Original": 0,
    "4K": 3840 * 2160,
    "2K": 2560 * 1440,
    "1080p": 1920 * 1080,
    "720p": 1280 * 720,
}
VIDEO_ENCODERS = ["ffmpeg", "opencv"]
DEFAULT_FFMPEG_CODEC = "libx264"
DEFAULT_FFMPEG_CRF = 18
DEFAULT_FFMPEG_PRESET = "medium"
TEMP_OUTPUT_DIR = Path(tempfile.gettempdir()).resolve() / "colorfm_l_video"

CUSTOM_CSS = """
#col-container { max-width: 1180px; margin: 0 auto; }
.status-box textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
"""

_MODEL: torch.nn.Module | None = None
_MODEL_DEVICE: torch.device | None = None
_MODEL_KEY: tuple[Any, ...] | None = None
_MODEL_LOCK = threading.Lock()

_KERNEL_CACHE: dict[str, dict[str, Any]] = {}
_KERNEL_LOCK = threading.Lock()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_cv2() -> None:
    if cv2 is None:
        raise gr.Error(
            "opencv-python-headless is required for video processing. "
            "Install it with `pip install opencv-python-headless`."
        )


def resolve_device(choice: str) -> torch.device:
    choice = str(choice or "auto").lower()
    if choice == "auto":
        return get_device()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise gr.Error("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise gr.Error("MPS was requested, but it is not available.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise gr.Error(f"Unsupported device: {choice}")


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    choice = str(choice or "auto").lower()
    if choice == "auto":
        return torch.float32
    if device.type == "cpu":
        return torch.float32
    if choice == "fp32":
        return torch.float32
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise gr.Error("BF16 was requested, but the selected device does not support it.")
        return torch.bfloat16
    raise gr.Error(f"Unsupported dtype: {choice}")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def clear_kernel_cache() -> None:
    with _KERNEL_LOCK:
        _KERNEL_CACHE.clear()


def clear_model() -> None:
    global _MODEL, _MODEL_DEVICE, _MODEL_KEY
    clear_kernel_cache()
    _MODEL = None
    _MODEL_DEVICE = None
    _MODEL_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("The ColorFM-L weight file is not a state dict.")
    return checkpoint


def load_model(
    checkpoint: str | Path | None = None,
    device_choice: str = "auto",
    dtype_choice: str = "auto",
) -> tuple[torch.nn.Module, torch.device]:
    global _MODEL, _MODEL_DEVICE, _MODEL_KEY

    checkpoint_path = resolve_repo_path(checkpoint or WEIGHTS_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ColorFM-L weights were not found at {checkpoint_path}. "
            "Choose a checkpoint in Settings or set COLORFM_L_WEIGHTS."
        )

    device = resolve_device(device_choice)
    dtype = resolve_dtype(dtype_choice, device)
    checkpoint_stat = checkpoint_path.stat()
    model_key = (
        str(checkpoint_path),
        checkpoint_stat.st_mtime_ns,
        checkpoint_stat.st_size,
        str(device),
        dtype,
    )

    if _MODEL is not None and _MODEL_DEVICE is not None and _MODEL_KEY == model_key:
        return _MODEL, _MODEL_DEVICE

    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_DEVICE is not None and _MODEL_KEY == model_key:
            return _MODEL, _MODEL_DEVICE

        clear_model()
        model = ColorFM_L(cfg=None)
        model.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
        model.to(device=device, dtype=dtype).eval()

        _MODEL = model
        _MODEL_DEVICE = device
        _MODEL_KEY = model_key
        return model, device


def model_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return model.get_params()


def swish(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def expand_kernel(kernel: torch.Tensor, batch_size: int) -> torch.Tensor:
    if kernel.shape[0] == batch_size:
        return kernel
    return kernel.expand(batch_size, -1, -1)


def run_kernel_stack(
    chunk: torch.Tensor,
    params: dict[str, torch.Tensor],
    suffix: str,
    t_value: float,
) -> torch.Tensor:
    batch_size = chunk.shape[0]
    t = torch.full(chunk.shape[:-1] + (1,), t_value, device=chunk.device, dtype=chunk.dtype)
    v = torch.cat([chunk, t], dim=-1)
    v = swish(torch.bmm(v, expand_kernel(params[f"kernel1_{suffix}"], batch_size)))
    v = swish(torch.bmm(v, expand_kernel(params[f"kernel2_{suffix}"], batch_size)))
    v = swish(torch.bmm(v, expand_kernel(params[f"kernel3_{suffix}"], batch_size)))
    return torch.bmm(v, expand_kernel(params[f"kernel4_{suffix}"], batch_size))


def apply_cached_kernel(
    frame_tensor: torch.Tensor,
    params: dict[str, torch.Tensor],
    strength: float,
    chunk_pixels: int,
) -> torch.Tensor:
    source = frame_tensor
    batch_size, channels, height, width = frame_tensor.shape
    x = frame_tensor.view(batch_size, channels, height * width).permute(0, 2, 1).contiguous()
    results = []

    chunk_pixels = max(1, int(chunk_pixels))
    for start in range(0, x.shape[1], chunk_pixels):
        chunk = x[:, start:start + chunk_pixels, :]
        chunk = chunk + run_kernel_stack(chunk, params, "content", 0.0)
        chunk = chunk - run_kernel_stack(chunk, params, "style", 1.0)
        results.append(chunk)

    output = torch.cat(results, dim=1)
    output = output.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)
    strength = float(strength)
    if strength < 1.0:
        output = source * (1.0 - strength) + output * strength
    return output.clamp(0, 1)


def video_path_from_value(video_value: Any) -> Path:
    if video_value is None:
        raise gr.Error("Please upload a video.")

    if isinstance(video_value, (str, Path)):
        video_path = Path(video_value).expanduser()
        if not video_path.is_file():
            raise gr.Error(f"Uploaded video was not found: {video_path}")
        return video_path.resolve()

    if isinstance(video_value, dict):
        for key in ("video", "path", "name"):
            value = video_value.get(key)
            if isinstance(value, dict):
                value = value.get("path") or value.get("name")
            if value:
                video_path = Path(value).expanduser()
                if not video_path.is_file():
                    raise gr.Error(f"Uploaded video was not found: {video_path}")
                return video_path.resolve()

    raise gr.Error("Could not read uploaded video path.")


def resize_image_to_square(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    pil_image = pil_image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(pil_image)


def image_to_tensor_batch(
    frames: list[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device=device, dtype=dtype)


def tensor_to_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    frames = tensor.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255.0 + 0.5).astype(np.uint8)
    return [frame for frame in frames]


def output_size_for_video(width: int, height: int, resolution_choice: str) -> tuple[int, int]:
    max_pixels = RESOLUTION_LIMITS.get(str(resolution_choice), 0)
    if max_pixels <= 0 or width * height <= max_pixels:
        out_width, out_height = width, height
    else:
        scale = math.sqrt(max_pixels / float(width * height))
        out_width = max(2, int(width * scale))
        out_height = max(2, int(height * scale))

    if out_width % 2:
        out_width -= 1
    if out_height % 2:
        out_height -= 1
    return max(2, out_width), max(2, out_height)


def get_video_meta(video_path: Path) -> dict[str, Any]:
    require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or not math.isfinite(fps):
        fps = 30.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if frame_count > 0 else 0.0,
    }


def read_video_frame(video_value: Any, frame_index: int, resolution_choice: str) -> tuple[np.ndarray, dict[str, Any]]:
    require_cv2()
    video_path = video_path_from_value(video_value)
    meta = get_video_meta(video_path)
    frame_count = int(meta["frame_count"])
    if frame_count > 0:
        frame_index = max(0, min(int(frame_index), frame_count - 1))
    else:
        frame_index = max(0, int(frame_index))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise gr.Error(f"Could not read frame {frame_index}.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_width, out_height = output_size_for_video(frame_rgb.shape[1], frame_rgb.shape[0], resolution_choice)
    if (out_width, out_height) != (frame_rgb.shape[1], frame_rgb.shape[0]):
        frame_rgb = cv2.resize(frame_rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)

    meta["frame_index"] = frame_index
    meta["preview_width"] = out_width
    meta["preview_height"] = out_height
    return frame_rgb, meta


def select_reference_frame(
    video_value: Any,
    frame_index: int,
    resolution_choice: str,
) -> tuple[np.ndarray, int, None, str]:
    frame, meta = read_video_frame(video_value, int(frame_index or 0), resolution_choice)
    selected_time = int(meta["frame_index"]) / float(meta["fps"])
    status = (
        f"selected: {selected_time:.3f}s / frame {meta['frame_index']}\n"
        f"video: {meta['width']}x{meta['height']}, {meta['fps']:.3f} fps\n"
        f"preview size: {meta['preview_width']}x{meta['preview_height']}\n"
        "generating the transfer preview..."
    )
    return frame, int(meta["frame_index"]), None, status


def trim_kernel_cache(max_items: int = 4) -> None:
    if len(_KERNEL_CACHE) <= max_items:
        return
    ordered = sorted(_KERNEL_CACHE.items(), key=lambda item: item[1]["created_at"])
    for key, _ in ordered[:-max_items]:
        _KERNEL_CACHE.pop(key, None)


def store_kernel_cache(
    params: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    checkpoint: str,
    frame_index: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    cache_id = uuid.uuid4().hex
    cache_entry: dict[str, Any] = {
        "params": params,
        "device": device,
        "dtype": dtype,
        "checkpoint": str(resolve_repo_path(checkpoint)),
        "frame_index": int(frame_index),
        "created_at": time.time(),
    }
    if metadata:
        cache_entry.update(metadata)

    with _KERNEL_LOCK:
        _KERNEL_CACHE[cache_id] = cache_entry
        trim_kernel_cache()
    return cache_id


def cache_kernel(
    video_value: Any,
    style_image: np.ndarray | None,
    frame_index: int,
    resolution_choice: str,
    checkpoint: str,
    device_choice: str,
    dtype_choice: str,
) -> tuple[np.ndarray | None, str, str]:
    if style_image is None:
        raise gr.Error("Please upload a style/reference image.")

    start = time.time()
    reference_frame, meta = read_video_frame(video_value, frame_index, resolution_choice)
    checkpoint_path = resolve_repo_path(checkpoint or WEIGHTS_PATH)
    model, device = load_model(checkpoint_path, device_choice, dtype_choice)
    dtype = resolve_dtype(dtype_choice, device)

    kernel_content = resize_image_to_square(reference_frame, KERNEL_IMAGE_SIZE)
    kernel_style = resize_image_to_square(style_image, KERNEL_IMAGE_SIZE)
    content_tensor = image_to_tensor_batch([kernel_content], device, dtype)
    style_tensor = image_to_tensor_batch([kernel_style], device, dtype)

    try:
        with _MODEL_LOCK, torch.inference_mode():
            model(content_tensor, style_tensor)
            raw_params = model_params(model)
            params = {
                key: value.detach().to(device=device, dtype=dtype).contiguous()
                for key, value in raw_params.items()
                if torch.is_tensor(value)
            }
            preview_tensor = image_to_tensor_batch([reference_frame], device, dtype)
            preview_tensor = apply_cached_kernel(
                preview_tensor,
                params,
                strength=1.0,
                chunk_pixels=1024 * 1024,
            )
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error("CUDA out of memory while caching kernels. Try 720p/CPU or fp16.") from exc

    cache_id = store_kernel_cache(
        params=params,
        device=device,
        dtype=dtype,
        checkpoint=str(checkpoint_path),
        frame_index=meta["frame_index"],
    )

    elapsed = time.time() - start
    preview = tensor_to_frames(preview_tensor)[0]
    status = (
        f"kernel cached in {elapsed:.2f}s\n"
        f"cache id: {cache_id[:8]}\n"
        f"device: {device}\n"
        f"dtype: {str(dtype).replace('torch.', '')}\n"
        f"reference frame: {meta['frame_index']}\n"
        "preview ready; render the full video after confirming this frame"
    )
    return preview, cache_id, status


def write_metadata(path: Path, params: dict[str, Any]) -> None:
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_temp_outputs(max_age_seconds: int = UPLOAD_CACHE_MAX_AGE) -> None:
    if not TEMP_OUTPUT_DIR.is_dir():
        return

    cutoff = time.time() - max(1, int(max_age_seconds))
    for path in TEMP_OUTPUT_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def format_fps(fps: float) -> str:
    if abs(fps - round(fps)) < 1e-6:
        return str(int(round(fps)))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def find_ffmpeg(ffmpeg_binary: str | None = None) -> str | None:
    candidates = []
    if ffmpeg_binary:
        candidates.append(Path(ffmpeg_binary).expanduser())

    env_binary = os.getenv("FFMPEG_BINARY")
    if env_binary:
        candidates.append(Path(env_binary).expanduser())

    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(Path(which_path))

    env_root = Path(sys.executable).resolve().parent
    candidates.extend([
        env_root / "ffmpeg.exe",
        env_root / "Library" / "bin" / "ffmpeg.exe",
        env_root / "Scripts" / "ffmpeg.exe",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return None


def resolve_ffmpeg(ffmpeg_binary: str | None = None) -> str:
    ffmpeg_path = find_ffmpeg(ffmpeg_binary)
    if ffmpeg_path is not None:
        return ffmpeg_path
    raise gr.Error(
        "ffmpeg was selected, but ffmpeg.exe was not found. "
        "Install it with `conda install -n colorfm -c conda-forge ffmpeg`, "
        "or paste the ffmpeg.exe path in FFmpeg Path."
    )


def default_video_encoder(ffmpeg_binary: str | None = None) -> str:
    return "ffmpeg" if find_ffmpeg(ffmpeg_binary) is not None else "opencv"


class OpenCVVideoWriter:
    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        require_cv2()
        self.codec = None
        self.writer = None

        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                self.writer = writer
                self.codec = codec
                break
            writer.release()

        if self.writer is None:
            raise gr.Error(f"Could not create output video: {output_path}")
        self.description = f"opencv/{self.codec}"

    def write_rgb_frame(self, frame_rgb: np.ndarray) -> None:
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


class FFmpegPipeWriter:
    def __init__(
        self,
        input_video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: int,
        preset: str,
        copy_audio: bool,
        ffmpeg_binary: str | None,
    ):
        ffmpeg_path = resolve_ffmpeg(ffmpeg_binary)
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            format_fps(fps),
            "-i",
            "pipe:0",
        ]

        if copy_audio:
            command.extend(["-i", str(input_video_path), "-map", "0:v:0", "-map", "1:a?"])
        else:
            command.extend(["-map", "0:v:0"])

        command.extend(["-c:v", codec])
        if codec == "h264_nvenc":
            command.extend(["-preset", preset, "-cq:v", str(int(crf)), "-b:v", "0"])
        else:
            command.extend(["-preset", preset, "-crf", str(int(crf))])

        command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        if copy_audio:
            command.extend(["-c:a", "copy", "-shortest"])
        else:
            command.append("-an")

        command.append(str(output_path))
        self.command = command
        self.description = f"ffmpeg/{codec}"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise gr.Error("Could not open ffmpeg stdin pipe.")

    def write_rgb_frame(self, frame_rgb: np.ndarray) -> None:
        try:
            frame_rgb = np.ascontiguousarray(frame_rgb)
            self.process.stdin.write(frame_rgb.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise gr.Error("ffmpeg pipe closed while writing frames.") from exc

    def close(self) -> None:
        stderr = b""
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise gr.Error(f"ffmpeg failed with exit code {return_code}.\n{message[-4000:]}")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()


def create_video_writer(
    encoder: str,
    input_video_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    ffmpeg_codec: str,
    ffmpeg_crf: int,
    ffmpeg_preset: str,
    copy_audio: bool,
    ffmpeg_binary: str | None,
) -> OpenCVVideoWriter | FFmpegPipeWriter:
    if encoder == "opencv":
        return OpenCVVideoWriter(output_path, width, height, fps)
    return FFmpegPipeWriter(
        input_video_path=input_video_path,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        codec=ffmpeg_codec,
        crf=ffmpeg_crf,
        preset=ffmpeg_preset,
        copy_audio=copy_audio,
        ffmpeg_binary=ffmpeg_binary,
    )


def render_video(
    video_value: Any,
    cache_id: str | None,
    resolution_choice: str,
    strength: float,
    max_frames: int,
    batch_size: int,
    chunk_megapixels: float,
    video_encoder: str,
    save_output: bool,
    output_dir: str,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[str | None, str]]:
    require_cv2()
    if not cache_id:
        raise gr.Error("Please cache a kernel first.")

    with _KERNEL_LOCK:
        cache = _KERNEL_CACHE.get(cache_id)
    if cache is None:
        raise gr.Error("Cached kernel was not found. Please cache it again.")

    video_path = video_path_from_value(video_value)
    meta = get_video_meta(video_path)
    output_width, output_height = output_size_for_video(meta["width"], meta["height"], resolution_choice)
    fps = float(meta["fps"])
    total_frames = int(meta["frame_count"])
    max_frames = int(max_frames or 0)
    frame_limit = total_frames if total_frames > 0 else max_frames
    if max_frames > 0 and frame_limit > 0:
        frame_limit = min(frame_limit, max_frames)

    if save_output:
        if not str(output_dir).strip():
            raise gr.Error("Output Dir is required when Save Output is enabled.")
        output_root = resolve_repo_path(output_dir)
    else:
        cleanup_temp_outputs()
        output_root = TEMP_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output_path = output_root / f"{stamp}_colorfm_l_video.mp4"
    video_encoder = str(video_encoder)
    if video_encoder not in VIDEO_ENCODERS:
        raise gr.Error(f"Unsupported video encoder: {video_encoder}")
    ffmpeg_codec = DEFAULT_FFMPEG_CODEC
    ffmpeg_crf = DEFAULT_FFMPEG_CRF
    ffmpeg_preset = DEFAULT_FFMPEG_PRESET
    copy_audio = video_encoder == "ffmpeg"
    ffmpeg_binary = DEFAULT_FFMPEG_PATH

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    writer = create_video_writer(
        encoder=video_encoder,
        input_video_path=video_path,
        output_path=output_path,
        width=output_width,
        height=output_height,
        fps=fps,
        ffmpeg_codec=ffmpeg_codec,
        ffmpeg_crf=ffmpeg_crf,
        ffmpeg_preset=ffmpeg_preset,
        copy_audio=bool(copy_audio),
        ffmpeg_binary=ffmpeg_binary,
    )
    params = cache["params"]
    device = cache["device"]
    dtype = cache["dtype"]
    batch_size = max(1, int(batch_size))
    chunk_pixels = max(1, int(float(chunk_megapixels) * 1024 * 1024))
    processed = 0
    batch_frames: list[np.ndarray] = []
    start = time.time()

    yield None, "rendering started..."

    def flush_batch() -> None:
        nonlocal batch_frames
        if not batch_frames:
            return
        with torch.inference_mode():
            frame_tensor = image_to_tensor_batch(batch_frames, device, dtype)
            output_tensor = apply_cached_kernel(frame_tensor, params, strength, chunk_pixels)
            output_frames = tensor_to_frames(output_tensor)
        for output_frame in output_frames:
            writer.write_rgb_frame(output_frame)
        batch_frames = []

    try:
        while True:
            if frame_limit > 0 and processed >= frame_limit:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if (output_width, output_height) != (frame_rgb.shape[1], frame_rgb.shape[0]):
                frame_rgb = cv2.resize(frame_rgb, (output_width, output_height), interpolation=cv2.INTER_AREA)

            batch_frames.append(frame_rgb)
            processed += 1
            if len(batch_frames) >= batch_size:
                flush_batch()

            if frame_limit > 0:
                progress(processed / frame_limit, desc=f"Rendering frame {processed}/{frame_limit}")
            else:
                progress(0, desc=f"Rendering frame {processed}")

        flush_batch()
        yield None, f"finalizing video after {processed} frames..."
    except torch.cuda.OutOfMemoryError as exc:
        if hasattr(writer, "abort"):
            writer.abort()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error("CUDA out of memory while rendering video. Reduce resolution, batch size, or chunk size.") from exc
    except Exception:
        if hasattr(writer, "abort"):
            writer.abort()
        raise
    finally:
        cap.release()

    writer.close()

    elapsed = max(time.time() - start, 1e-6)
    output_metadata = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "checkpoint": cache["checkpoint"],
        "kernel_frame_index": cache["frame_index"],
        "resolution": resolution_choice,
        "input_size": [meta["width"], meta["height"]],
        "output_size": [output_width, output_height],
        "source_fps": fps,
        "frames": processed,
        "strength": strength,
        "batch_size": batch_size,
        "chunk_megapixels": chunk_megapixels,
        "video_encoder": video_encoder,
        "ffmpeg_codec": ffmpeg_codec if video_encoder == "ffmpeg" else None,
        "ffmpeg_crf": ffmpeg_crf if video_encoder == "ffmpeg" else None,
        "ffmpeg_preset": ffmpeg_preset if video_encoder == "ffmpeg" else None,
        "ffmpeg_binary": str(ffmpeg_binary) if video_encoder == "ffmpeg" and ffmpeg_binary else None,
        "copy_audio": bool(copy_audio) if video_encoder == "ffmpeg" else False,
        "saved": bool(save_output),
        "device": str(device),
        "dtype": str(dtype),
    }
    for key in (
        "reference_sdr_video",
        "reference_enhanced_video",
        "reference_time_seconds",
        "reference_sdr_frame_index",
        "reference_enhanced_frame_index",
    ):
        if key in cache:
            output_metadata[key] = cache[key]
    write_metadata(output_path, output_metadata)

    status = (
        f"rendered {processed} frames in {elapsed:.2f}s\n"
        f"throughput: {processed / elapsed:.2f} fps\n"
        f"output: {output_width}x{output_height}, {fps:.3f} fps\n"
        f"encoder: {writer.description}\n"
        f"audio: {'copied when present' if video_encoder == 'ffmpeg' and copy_audio else 'not preserved'}"
    )
    if save_output:
        status += f"\nsaved: {output_path.resolve()}"
    else:
        status += "\nsaved: no (temporary preview file only)"
    if isinstance(writer, OpenCVVideoWriter) and writer.codec == "mp4v":
        status += "\nweb playback: mp4v may not preview in browsers; download it or use ffmpeg/libx264"
    output_path_str = str(output_path.resolve())
    yield output_path_str, status


def reset_video_selection() -> tuple[None, int, None, None, str]:
    return None, 0, None, None, "Video loaded. Enter a reference frame number."


def reset_transfer_preview() -> tuple[None, None, str]:
    return None, None, "Reference image changed. Generate the transfer preview again."


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="ColorFM-L Video",
        delete_cache=(UPLOAD_CACHE_CLEANUP_INTERVAL, UPLOAD_CACHE_MAX_AGE),
    ) as demo:
        kernel_state = gr.State(None)
        selected_frame_state = gr.State(0)

        with gr.Column(elem_id="col-container"):
            gr.Markdown("# 🎬 ColorFM-L Video Color Transfer")

            with gr.Accordion(label="📖 How to Use", open=False):
                gr.Markdown(
                    """
1. **Upload the inputs:** upload the content video on the left and the color reference image on the right.
2. **Choose a reference frame:** enter the target frame number in Settings, then click **📌 Use Selected Frame & Generate Preview**.
3. **Review and render:** check the selected frame and transfer preview, adjust the strength and resolution, then click **🚀 Render Video**.
4. **Choose whether to save:** enable **💾 Save Output** to write the video and metadata permanently to Output Dir. When disabled, only a temporary preview file is created.
                    """
                )

            with gr.Row():
                video_input = gr.Video(
                    label="🎞️ Content Video",
                    sources=["upload"],
                    format=None,
                    include_audio=True,
                    streaming=False,
                    elem_id="content-video",
                    height=320,
                )
                style_input = gr.Image(
                    label="🎨 Style / Reference Image",
                    type="numpy",
                    height=320,
                    image_mode="RGB",
                )

            with gr.Group():
                with gr.Row():
                    frame_index = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        label="🎞️ Reference Frame",
                    )
                    resolution = gr.Radio(
                        choices=list(RESOLUTION_LIMITS),
                        value="1080p",
                        label="📐 Max Resolution",
                    )
                    strength = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.05,
                        label="🎚️ Transfer Strength",
                    )

            select_frame_button = gr.Button(
                "📌 Use Selected Frame & Generate Preview",
                variant="primary",
            )

            with gr.Row():
                reference_preview = gr.Image(
                    label="📍 Selected Reference Frame",
                    type="numpy",
                    interactive=False,
                    height=300,
                )
                kernel_preview = gr.Image(
                    label="✨ Transfer Preview",
                    type="numpy",
                    interactive=False,
                    height=300,
                )

            with gr.Accordion(label="⚙️ Advanced Settings", open=False):
                with gr.Row():
                    video_encoder = gr.Radio(
                        VIDEO_ENCODERS,
                        value=default_video_encoder(DEFAULT_FFMPEG_PATH),
                        label="🎞️ Video Encoder",
                    )
                    save_output = gr.Checkbox(
                        value=False,
                        label="💾 Save Output",
                        info="Save the video and metadata permanently to Output Dir.",
                    )

                with gr.Row():
                    max_frames = gr.Number(
                        value=0,
                        precision=0,
                        label="🔢 Max Frames",
                        info="0 processes the complete video.",
                    )
                    batch_size = gr.Slider(
                        1,
                        8,
                        value=1,
                        step=1,
                        label="📦 Batch Size",
                    )
                    chunk_megapixels = gr.Slider(
                        0.25,
                        4.0,
                        value=1.0,
                        step=0.25,
                        label="🧩 Chunk MPixels",
                    )

                with gr.Row():
                    checkpoint = gr.Textbox(
                        value=str(Path(os.getenv("COLORFM_L_WEIGHTS", "checkpoints/colorfm_l.pth"))),
                        label="🧠 Checkpoint",
                    )
                    output_dir = gr.Textbox(
                        value=str(DEFAULT_OUTPUT_DIR),
                        label="📁 Output Dir",
                        info="Used only when Save Output is enabled.",
                    )

                with gr.Row():
                    device = gr.Dropdown(
                        DEVICE_CHOICES,
                        value=os.getenv("COLORFM_L_DEVICE", "auto"),
                        label="🖥️ Device",
                    )
                    dtype = gr.Dropdown(
                        DTYPE_CHOICES,
                        value=os.getenv("COLORFM_L_DTYPE", "auto"),
                        label="🔢 Dtype",
                        info="Auto preserves the original FP32 inference behavior.",
                    )

            render_button = gr.Button("🚀 Render Video", variant="primary")

            output_video = gr.Video(label="🎬 Result Video", interactive=False)
            status = gr.Textbox(
                label="📋 Status",
                lines=7,
                interactive=False,
                elem_classes="status-box",
            )

        selected_frame_event = select_frame_button.click(
            fn=select_reference_frame,
            inputs=[video_input, frame_index, resolution],
            outputs=[reference_preview, selected_frame_state, kernel_preview, status],
        )
        selected_frame_event.then(
            fn=cache_kernel,
            inputs=[
                video_input,
                style_input,
                selected_frame_state,
                resolution,
                checkpoint,
                device,
                dtype,
            ],
            outputs=[kernel_preview, kernel_state, status],
        )

        video_input.change(
            fn=reset_video_selection,
            outputs=[
                reference_preview,
                selected_frame_state,
                kernel_preview,
                kernel_state,
                status,
            ],
        )
        style_input.change(
            fn=reset_transfer_preview,
            outputs=[kernel_preview, kernel_state, status],
        )

        render_button.click(
            fn=render_video,
            inputs=[
                video_input,
                kernel_state,
                resolution,
                strength,
                max_frames,
                batch_size,
                chunk_megapixels,
                video_encoder,
                save_output,
                output_dir,
            ],
            outputs=[output_video, status],
        )

    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CUSTOM_CSS)
