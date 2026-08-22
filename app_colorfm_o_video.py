from __future__ import annotations

import gc
import hashlib
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
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

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
from omegaconf import OmegaConf
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from datasets.color_pairs import FlowDataset
from solvers.colorfm_o_solver import FlowSolver, fit_flow_solver


REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "configs" / "colorfm_o.yaml"
DEFAULT_CONFIG = OmegaConf.load(CONFIG_PATH)
SEGMENTATION_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
DEFAULT_OUTPUT_DIR = Path(os.getenv("COLORFM_O_VIDEO_OUTPUT_DIR", "outputs/colorfm_o_video"))
DEFAULT_FFMPEG_PATH = os.getenv("FFMPEG_BINARY", "")
UPLOAD_CACHE_CLEANUP_INTERVAL = 60 * 60
UPLOAD_CACHE_MAX_AGE = 24 * 60 * 60

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
TEMP_OUTPUT_DIR = Path(tempfile.gettempdir()).resolve() / "colorfm_o_video"

CUSTOM_CSS = """
#col-container { max-width: 1180px; margin: 0 auto; }
.status-box textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
"""

_FIT_LOCK = threading.Lock()
_FLOW_LOCK = threading.Lock()
_FLOW_CACHE: dict[str, dict[str, Any]] = {}
_MAX_CACHED_FLOWS = 2


class _CapturingFlowDataset(FlowDataset):
    """Expose processed segmentation maps without changing FlowDataset."""

    content_segmentation: np.ndarray | None = None
    style_segmentation: np.ndarray | None = None

    def process_seg_map(self, image_seg, style_seg, seg_mode):
        content_segmentation, style_segmentation = super().process_seg_map(
            image_seg,
            style_seg,
            seg_mode,
        )
        if seg_mode:
            self.content_segmentation = (
                content_segmentation.detach().squeeze().cpu().numpy().astype(np.uint8)
            )
            self.style_segmentation = (
                style_segmentation.detach().squeeze().cpu().numpy().astype(np.uint8)
            )
        return content_segmentation, style_segmentation


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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if np.issubdtype(image.dtype, np.floating) and max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def video_path_from_value(video_value: Any) -> Path:
    if video_value is None:
        raise gr.Error("Please upload a content video.")

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

    raise gr.Error("Could not read the uploaded video path.")


def get_video_meta(video_path: Path) -> dict[str, Any]:
    require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if fps <= 0 or not math.isfinite(fps):
        fps = 30.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if frame_count > 0 else 0.0,
    }


def output_size_for_video(
    width: int,
    height: int,
    resolution_choice: str,
) -> tuple[int, int]:
    max_pixels = RESOLUTION_LIMITS.get(str(resolution_choice), 0)
    if max_pixels <= 0 or width * height <= max_pixels:
        output_width, output_height = width, height
    else:
        scale = math.sqrt(max_pixels / float(width * height))
        output_width = max(2, int(width * scale))
        output_height = max(2, int(height * scale))

    if output_width % 2:
        output_width -= 1
    if output_height % 2:
        output_height -= 1
    return max(2, output_width), max(2, output_height)


def resize_rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    downsample = frame.shape[1] > width or frame.shape[0] > height
    interpolation = cv2.INTER_AREA if downsample else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def read_video_frame(
    video_value: Any,
    frame_index: int,
    resolution_choice: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    require_cv2()
    video_path = video_path_from_value(video_value)
    meta = get_video_meta(video_path)
    frame_count = int(meta["frame_count"])
    if frame_count > 0:
        frame_index = max(0, min(int(frame_index), frame_count - 1))
    else:
        frame_index = max(0, int(frame_index))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise gr.Error(f"Could not read frame {frame_index}.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    output_width, output_height = output_size_for_video(
        frame_rgb.shape[1],
        frame_rgb.shape[0],
        resolution_choice,
    )
    frame_rgb = resize_rgb_frame(frame_rgb, output_width, output_height)
    meta["frame_index"] = frame_index
    meta["preview_width"] = output_width
    meta["preview_height"] = output_height
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
        "fitting the online flow and generating the preview..."
    )
    return frame, int(meta["frame_index"]), None, status


def segmentation_palette() -> list[int]:
    palette: list[int] = []
    for label in range(256):
        value = label
        red = green = blue = 0
        for shift in range(8):
            red |= ((value >> 0) & 1) << (7 - shift)
            green |= ((value >> 1) & 1) << (7 - shift)
            blue |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
        palette.extend((red, green, blue))
    return palette


def save_segmentation_map(
    segmentation: np.ndarray,
    path: Path,
    size: tuple[int, int],
) -> None:
    mask = Image.fromarray(np.asarray(segmentation, dtype=np.uint8), mode="P")
    mask.putpalette(segmentation_palette())
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    mask.save(path)


def load_segmentation_model(device: torch.device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        SEGMENTATION_MODEL_ID,
    ).to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(
        SEGMENTATION_MODEL_ID,
    )
    return model, processor


def build_color_pairs_with_segmentations(
    content: np.ndarray,
    style: np.ndarray,
    device: torch.device,
    use_segmentation: bool,
) -> tuple[torch.Tensor, tuple[np.ndarray, np.ndarray] | None]:
    segmentation_model = None
    segmentation_processor = None
    if use_segmentation:
        segmentation_model, segmentation_processor = load_segmentation_model(device)

    with tempfile.TemporaryDirectory(prefix="colorfm_o_video_fit_") as temp_dir:
        temp_path = Path(temp_dir)
        content_path = temp_path / "content.png"
        style_path = temp_path / "style.png"
        Image.fromarray(content).save(content_path)
        Image.fromarray(style).save(style_path)

        data_config = SimpleNamespace(
            x_0=str(content_path),
            x_1=str(style_path),
            path="inference",
            batch_size=4096,
            num_workers=0,
        )
        dataset = _CapturingFlowDataset(
            data_config,
            device=device,
            SegModel=segmentation_model,
            feature_extractor=segmentation_processor,
            full=True,
            seg_mode=use_segmentation,
        )

    del segmentation_model, segmentation_processor
    segmentation_maps = None
    if use_segmentation:
        if dataset.content_segmentation is None or dataset.style_segmentation is None:
            raise RuntimeError("Semantic segmentation maps were not captured.")
        segmentation_maps = (
            dataset.content_segmentation,
            dataset.style_segmentation,
        )
    return dataset.pairs, segmentation_maps


def make_fit_cache_key(
    content: np.ndarray,
    style: np.ndarray,
    fit_steps: int,
    use_segmentation: bool,
) -> str:
    digest = hashlib.sha256()
    for image in (content, style):
        contiguous = np.ascontiguousarray(image)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    digest.update(str(int(fit_steps)).encode())
    digest.update(str(bool(use_segmentation)).encode())
    digest.update(CONFIG_PATH.read_bytes())
    return digest.hexdigest()


def trim_flow_cache() -> None:
    while len(_FLOW_CACHE) > _MAX_CACHED_FLOWS:
        oldest_id = min(_FLOW_CACHE, key=lambda key: float(_FLOW_CACHE[key]["created_at"]))
        old_cache = _FLOW_CACHE.pop(oldest_id)
        old_cache["solver"].to("cpu")


def find_cached_flow(fit_key: str) -> tuple[str, dict[str, Any]] | None:
    with _FLOW_LOCK:
        for cache_id, cache in _FLOW_CACHE.items():
            if cache["fit_key"] == fit_key:
                return cache_id, cache
    return None


def store_flow_cache(cache: dict[str, Any]) -> str:
    cache_id = uuid.uuid4().hex
    with _FLOW_LOCK:
        _FLOW_CACHE[cache_id] = cache
        trim_flow_cache()
    return cache_id


def fit_flow_from_frame(
    content_frame: np.ndarray | None,
    style_image: np.ndarray | None,
    frame_index: int,
    fit_steps: int,
    sampling_steps: int,
    use_segmentation: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[np.ndarray, str, str]:
    if content_frame is None or style_image is None:
        raise gr.Error("Select a video frame and upload a style/reference image first.")
    fit_steps = int(fit_steps)
    sampling_steps = int(sampling_steps)
    if fit_steps <= 0:
        raise gr.Error("Fit steps must be greater than zero.")
    if sampling_steps <= 0:
        raise gr.Error("Sampling steps must be greater than zero.")

    with _FIT_LOCK:
        started = time.perf_counter()
        device = get_device()
        content = normalize_image(content_frame)
        style = normalize_image(style_image)
        fit_key = make_fit_cache_key(content, style, fit_steps, use_segmentation)
        cached = find_cached_flow(fit_key)
        cache_reused = cached is not None

        try:
            if cached is None:
                progress(0, desc="Building color pairs")
                pairs, segmentation_maps = build_color_pairs_with_segmentations(
                    content,
                    style,
                    device,
                    use_segmentation,
                )
                config = OmegaConf.load(CONFIG_PATH)
                config.train.total_steps = fit_steps
                solver = fit_flow_solver(
                    pairs=pairs,
                    cfg=config,
                    progress_callback=lambda step, total: progress(
                        (step, total),
                        desc=f"Optimizing ({step}/{total})",
                    ),
                )
                solver = solver.to(device).eval()
                cache = {
                    "solver": solver,
                    "device": device,
                    "fit_key": fit_key,
                    "fit_steps": fit_steps,
                    "frame_index": int(frame_index),
                    "use_segmentation": bool(use_segmentation),
                    "segmentations": segmentation_maps,
                    "content_size": (content.shape[1], content.shape[0]),
                    "style_size": (style.shape[1], style.shape[0]),
                    "created_at": time.time(),
                }
                cache_id = store_flow_cache(cache)
            else:
                cache_id, cache = cached
                solver = cache["solver"].to(device).eval()
                cache["device"] = device
                progress(0, desc="Reusing cached flow model")

            progress(1.0, desc="Generating transfer preview")
            preview = solver.transform_image(
                content,
                sampling_steps=sampling_steps,
                transfer_strength=1.0,
            )
            status = (
                f"flow ready in {time.perf_counter() - started:.2f}s\n"
                f"cache id: {cache_id[:8]}\n"
                f"device: {device}\n"
                f"fit model: {'cache reused' if cache_reused else 'optimized'}\n"
                f"fit steps: {fit_steps}\n"
                f"sampling steps: {sampling_steps}\n"
                f"semantic segmentation: {'on' if use_segmentation else 'off'}\n"
                f"reference frame: {int(frame_index)}"
            )
            return preview, cache_id, status
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(
                "CUDA out of memory while fitting the flow. Disable segmentation or use CPU."
            ) from exc
        except gr.Error:
            raise
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    candidates: list[Path] = []
    if ffmpeg_binary:
        candidates.append(Path(ffmpeg_binary).expanduser())
    env_binary = os.getenv("FFMPEG_BINARY")
    if env_binary:
        candidates.append(Path(env_binary).expanduser())
    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(Path(which_path))

    env_root = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            env_root / "ffmpeg.exe",
            env_root / "Library" / "bin" / "ffmpeg.exe",
            env_root / "Scripts" / "ffmpeg.exe",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def resolve_ffmpeg(ffmpeg_binary: str | None = None) -> str:
    ffmpeg_path = find_ffmpeg(ffmpeg_binary)
    if ffmpeg_path is not None:
        return ffmpeg_path
    raise gr.Error(
        "ffmpeg was selected, but ffmpeg.exe was not found. "
        "Install it with `conda install -n colorfm -c conda-forge ffmpeg` "
        "or set FFMPEG_BINARY."
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

    def abort(self) -> None:
        self.close()


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
    ) -> None:
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
        command.extend(["-c:v", codec, "-preset", preset, "-crf", str(int(crf))])
        command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        if copy_audio:
            command.extend(["-c:a", "copy", "-shortest"])
        else:
            command.append("-an")
        command.append(str(output_path))

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
            self.process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
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
    copy_audio: bool,
) -> OpenCVVideoWriter | FFmpegPipeWriter:
    if encoder == "opencv":
        return OpenCVVideoWriter(output_path, width, height, fps)
    return FFmpegPipeWriter(
        input_video_path=input_video_path,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        codec=DEFAULT_FFMPEG_CODEC,
        crf=DEFAULT_FFMPEG_CRF,
        preset=DEFAULT_FFMPEG_PRESET,
        copy_audio=copy_audio,
        ffmpeg_binary=DEFAULT_FFMPEG_PATH,
    )


def apply_flow_to_frames(
    solver: FlowSolver,
    frames: list[np.ndarray],
    sampling_steps: int,
    transfer_strength: float,
    chunk_pixels: int,
) -> list[np.ndarray]:
    source = np.stack(frames, axis=0).astype(np.float32) / 255.0
    pixels = torch.from_numpy(source.reshape(-1, 3))
    output = solver.transform_pixels(
        pixels,
        chunk_size=chunk_pixels,
        sampling_steps=int(sampling_steps),
        transfer_strength=float(transfer_strength),
    )
    output = output.reshape(source.shape).numpy()
    output = np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)
    return [frame for frame in output]


def write_metadata(path: Path, params: dict[str, Any]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def render_video(
    video_value: Any,
    cache_id: str | None,
    resolution_choice: str,
    transfer_strength: float,
    sampling_steps: int,
    max_frames: int,
    batch_size: int,
    chunk_megapixels: float,
    video_encoder: str,
    save_output: bool,
    save_segmentation: bool,
    output_dir: str,
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[str | None, str]]:
    require_cv2()
    if not cache_id:
        raise gr.Error("Select a reference frame and fit the online flow first.")
    with _FLOW_LOCK:
        cache = _FLOW_CACHE.get(cache_id)
    if cache is None:
        raise gr.Error("The cached flow was not found. Generate the preview again.")
    if not 0.0 <= float(transfer_strength) <= 1.0:
        raise gr.Error("Transfer strength must be between zero and one.")
    if int(sampling_steps) <= 0:
        raise gr.Error("Sampling steps must be greater than zero.")

    video_path = video_path_from_value(video_value)
    meta = get_video_meta(video_path)
    output_width, output_height = output_size_for_video(
        int(meta["width"]),
        int(meta["height"]),
        resolution_choice,
    )
    fps = float(meta["fps"])
    total_frames = int(meta["frame_count"])
    max_frames = max(0, int(max_frames or 0))
    frame_limit = total_frames if total_frames > 0 else max_frames
    if max_frames > 0 and frame_limit > 0:
        frame_limit = min(frame_limit, max_frames)
    if frame_limit <= 0:
        raise gr.Error("Could not determine the video frame count.")

    if save_output:
        if not str(output_dir).strip():
            raise gr.Error("Output Dir is required when Save Output is enabled.")
        output_root = resolve_repo_path(output_dir)
    else:
        cleanup_temp_outputs()
        output_root = TEMP_OUTPUT_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}-{uuid.uuid4().hex[:8]}"
    output_path = output_root / f"{stamp}_colorfm_o_video.mp4"

    video_encoder = str(video_encoder)
    if video_encoder not in VIDEO_ENCODERS:
        raise gr.Error(f"Unsupported video encoder: {video_encoder}")
    copy_audio = video_encoder == "ffmpeg"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise gr.Error(f"Could not open video: {video_path}")

    solver = cache["solver"].to(cache["device"]).eval()
    batch_size = max(1, int(batch_size))
    chunk_pixels = max(1, int(float(chunk_megapixels) * 1024 * 1024))
    batch_frames: list[np.ndarray] = []
    processed = 0
    writer: OpenCVVideoWriter | FFmpegPipeWriter | None = None
    started = time.perf_counter()

    yield None, "rendering started..."

    try:
        writer = create_video_writer(
            encoder=video_encoder,
            input_video_path=video_path,
            output_path=output_path,
            width=output_width,
            height=output_height,
            fps=fps,
            copy_audio=copy_audio,
        )

        def flush_batch() -> None:
            nonlocal batch_frames
            if not batch_frames:
                return
            output_frames = apply_flow_to_frames(
                solver,
                batch_frames,
                int(sampling_steps),
                float(transfer_strength),
                chunk_pixels,
            )
            for output_frame in output_frames:
                writer.write_rgb_frame(output_frame)
            batch_frames = []

        while processed < frame_limit:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = resize_rgb_frame(frame_rgb, output_width, output_height)
            batch_frames.append(frame_rgb)
            processed += 1
            if len(batch_frames) >= batch_size:
                flush_batch()
            progress(processed / frame_limit, desc=f"Rendering frame {processed}/{frame_limit}")

        flush_batch()
        if processed == 0:
            raise gr.Error("No frames were read from the video.")
        yield None, f"finalizing video after {processed} frames..."
        writer.close()
    except torch.cuda.OutOfMemoryError as exc:
        if writer is not None:
            writer.abort()
        if output_path.exists():
            output_path.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(
            "CUDA out of memory while rendering. Reduce resolution, batch size, or chunk size."
        ) from exc
    except Exception:
        if writer is not None:
            writer.abort()
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        capture.release()

    segmentation_files: dict[str, str] = {}
    segmentations = cache.get("segmentations")
    if save_output and save_segmentation and segmentations is not None:
        content_segmentation_path = output_path.with_name(
            f"{output_path.stem}_content_segmentation.png"
        )
        style_segmentation_path = output_path.with_name(
            f"{output_path.stem}_style_segmentation.png"
        )
        save_segmentation_map(
            segmentations[0],
            content_segmentation_path,
            cache["content_size"],
        )
        save_segmentation_map(
            segmentations[1],
            style_segmentation_path,
            cache["style_size"],
        )
        segmentation_files = {
            "content": str(content_segmentation_path.resolve()),
            "style": str(style_segmentation_path.resolve()),
        }

    elapsed = max(time.perf_counter() - started, 1e-6)
    metadata = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "config": str(CONFIG_PATH),
        "reference_frame": cache["frame_index"],
        "fit_steps": cache["fit_steps"],
        "sampling_steps": int(sampling_steps),
        "transfer_strength": float(transfer_strength),
        "semantic_segmentation": cache["use_segmentation"],
        "segmentation_maps": segmentation_files,
        "resolution": resolution_choice,
        "input_size": [meta["width"], meta["height"]],
        "output_size": [output_width, output_height],
        "source_fps": fps,
        "frames": processed,
        "batch_size": batch_size,
        "chunk_megapixels": float(chunk_megapixels),
        "video_encoder": video_encoder,
        "copy_audio": copy_audio,
        "saved": bool(save_output),
        "device": str(cache["device"]),
    }
    write_metadata(output_path, metadata)

    status = (
        f"rendered {processed} frames in {elapsed:.2f}s\n"
        f"throughput: {processed / elapsed:.2f} fps\n"
        f"output: {output_width}x{output_height}, {fps:.3f} fps\n"
        f"fit steps: {cache['fit_steps']}\n"
        f"sampling steps: {int(sampling_steps)}\n"
        f"semantic segmentation: {'on' if cache['use_segmentation'] else 'off'}\n"
        f"encoder: {writer.description}\n"
        f"audio: {'copied when present' if copy_audio else 'not preserved'}"
    )
    if save_output:
        status += f"\nsaved: {output_path.resolve()}"
        if segmentation_files:
            status += "\nsegmentation maps: saved"
        elif save_segmentation:
            status += "\nsegmentation maps: not saved (Semantic Segmentation is off)"
    else:
        status += "\nsaved: no (temporary preview file only)"
        if save_segmentation:
            status += "\nsegmentation maps: not saved (Save Output is off)"
    if isinstance(writer, OpenCVVideoWriter) and writer.codec == "mp4v":
        status += "\nweb playback: mp4v may not preview in browsers; download it or use ffmpeg/libx264"
    yield str(output_path.resolve()), status


def reset_video_selection() -> tuple[None, int, None, None, str]:
    return None, 0, None, None, "Video loaded. Enter a reference frame number."


def reset_transfer_preview() -> tuple[None, None, str]:
    return None, None, "Reference image changed. Generate the transfer preview again."


def reset_reference_selection(frame_index: int) -> tuple[None, int, None, None, str]:
    return (
        None,
        max(0, int(frame_index or 0)),
        None,
        None,
        "Reference frame settings changed. Generate the transfer preview again.",
    )


def reset_fit_preview(_value: Any) -> tuple[None, None, str]:
    return None, None, "Flow fitting settings changed. Generate the transfer preview again."


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="ColorFM-O Video",
        delete_cache=(UPLOAD_CACHE_CLEANUP_INTERVAL, UPLOAD_CACHE_MAX_AGE),
    ) as demo:
        flow_state = gr.State(None)
        selected_frame_state = gr.State(0)

        with gr.Column(elem_id="col-container"):
            gr.Markdown("# 🎬 ColorFM-O Video Color Transfer")

            with gr.Accordion(label="📖 How to Use", open=False):
                gr.Markdown(
                    """
1. **Upload the inputs:** upload the content video on the left and the color reference image on the right.
2. **Choose a reference frame:** enter its frame number, then click **📌 Use Selected Frame & Fit Flow**.
3. **Review and render:** check the selected frame and flow preview, adjust the settings, then click **🚀 Render Video**.
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
                    height=320,
                )
                style_input = gr.Image(
                    label="🎨 Style / Reference Image",
                    type="numpy",
                    image_mode="RGB",
                    height=320,
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
                    transfer_strength = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.05,
                        label="🎚️ Transfer Strength",
                        info="0 keeps the original colors; 1 follows the complete flow path.",
                    )

                with gr.Row():
                    fit_steps = gr.Slider(
                        1,
                        1000,
                        value=int(DEFAULT_CONFIG.train.total_steps),
                        step=1,
                        label="🔁 Fit Steps",
                    )
                    sampling_steps = gr.Slider(
                        1,
                        50,
                        value=round(1.0 / float(DEFAULT_CONFIG.inference.ode_step_size)),
                        step=1,
                        label="🪜 Sampling Steps",
                        info="More steps can improve ODE accuracy but increase inference time.",
                    )
                    use_segmentation = gr.Checkbox(
                        value=False,
                        label="🧩 Semantic Segmentation",
                    )

            fit_button = gr.Button("📌 Use Selected Frame & Fit Flow", variant="primary")

            with gr.Row():
                reference_preview = gr.Image(
                    label="📍 Selected Reference Frame",
                    type="numpy",
                    interactive=False,
                    height=300,
                )
                flow_preview = gr.Image(
                    label="✨ Flow Transfer Preview",
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
                    save_segmentation = gr.Checkbox(
                        value=False,
                        label="🧩 Save Segmentation Maps",
                        info="Used only when Save Output and Semantic Segmentation are enabled.",
                    )

                with gr.Row():
                    max_frames = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
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

                output_dir = gr.Textbox(
                    value=str(DEFAULT_OUTPUT_DIR),
                    label="📁 Output Dir",
                    info="Used only when Save Output is enabled.",
                )

            render_button = gr.Button("🚀 Render Video", variant="primary")
            output_video = gr.Video(label="🎬 Result Video", interactive=False)
            status = gr.Textbox(
                label="📋 Status",
                lines=9,
                interactive=False,
                elem_classes="status-box",
            )

        selected_frame_event = fit_button.click(
            fn=select_reference_frame,
            inputs=[video_input, frame_index, resolution],
            outputs=[reference_preview, selected_frame_state, flow_preview, status],
        )
        selected_frame_event.then(
            fn=fit_flow_from_frame,
            inputs=[
                reference_preview,
                style_input,
                selected_frame_state,
                fit_steps,
                sampling_steps,
                use_segmentation,
            ],
            outputs=[flow_preview, flow_state, status],
        )

        video_input.change(
            fn=reset_video_selection,
            outputs=[
                reference_preview,
                selected_frame_state,
                flow_preview,
                flow_state,
                status,
            ],
        )
        style_input.change(
            fn=reset_transfer_preview,
            outputs=[flow_preview, flow_state, status],
        )
        frame_index.change(
            fn=reset_reference_selection,
            inputs=frame_index,
            outputs=[
                reference_preview,
                selected_frame_state,
                flow_preview,
                flow_state,
                status,
            ],
        )
        resolution.change(
            fn=reset_reference_selection,
            inputs=frame_index,
            outputs=[
                reference_preview,
                selected_frame_state,
                flow_preview,
                flow_state,
                status,
            ],
        )
        fit_steps.change(
            fn=reset_fit_preview,
            inputs=fit_steps,
            outputs=[flow_preview, flow_state, status],
        )
        use_segmentation.change(
            fn=reset_fit_preview,
            inputs=use_segmentation,
            outputs=[flow_preview, flow_state, status],
        )

        render_button.click(
            fn=render_video,
            inputs=[
                video_input,
                flow_state,
                resolution,
                transfer_strength,
                sampling_steps,
                max_frames,
                batch_size,
                chunk_megapixels,
                video_encoder,
                save_output,
                save_segmentation,
                output_dir,
            ],
            outputs=[output_video, status],
        )

    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CUSTOM_CSS)
