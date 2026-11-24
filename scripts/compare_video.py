"""Compare two trained YOLOv8 checkpoints on the exact same video frames."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(ffmpeg, source, target, speed=1.0, fps=30.0):
    # Reinterpret the constant-rate input timestamps at the requested playback FPS.
    # No trimming, frame dropping or output synchronization duplicates are needed.
    command = [ffmpeg, "-y", "-loglevel", "error", "-r", str(fps * speed), "-i", str(source)]
    command += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-fps_mode", "passthrough", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--colab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=2.0)
    options = parser.parse_args()
    if options.playback_speed <= 0:
        parser.error("Playback speed must be positive")

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    options.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(options.video))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok or fps <= 0:
        raise RuntimeError("Invalid video metadata / empty video")
    height, width = first.shape[:2]
    metadata = {"input": str(options.video), "sha256": hash_file(options.video),
                "fps": fps, "width": width, "height": height,
                "metadata_frame_count": frame_count, "duration_seconds": frame_count / fps}
    (options.out / "video_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)

    # The contact sheet is an inspection artifact, not an annotated ground-truth dataset.
    sample_indices = np.linspace(0, max(0, frame_count - 1), 8).astype(int).tolist()
    thumb_w = 480
    thumb_h = int(height * thumb_w / width)
    sheet = Image.new("RGB", (thumb_w * 4, (thumb_h + 38) * 2), "#141b26")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    for n, index in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        x, y = (n % 4) * thumb_w, (n // 4) * (thumb_h + 38)
        thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((thumb_w, thumb_h))
        sheet.paste(thumb, (x, y + 38))
        draw.text((x + 10, y + 6), f"{index / fps:.2f} s", font=font, fill="white")
    sheet.save(options.out / "input_contact_sheet.jpg", quality=92)
    cap.release()
    if options.inspect_only:
        return

    import torch
    import ultralytics
    import imageio_ffmpeg
    from ultralytics import YOLO

    torch.set_num_threads(min(4, os.cpu_count() or 2))
    torch.manual_seed(0)
    metadata["inference_environment"] = {
        "device": "cpu", "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__, "imgsz": 640, "conf": 0.25, "iou": 0.7,
        "tracker": "bytetrack.yaml", "frame_stride": 1, "precision": "FP32",
        "note": "Training environment labels do not describe the inference device; both models run here on CPU."}
    models = [
        ("local_196_75", "LOCAL / 196 images / 75 epochs", options.local),
        ("colab_414_50", "COLAB / 414 images / 50 epochs", options.colab),
    ]
    colors = {"cup": (0, 210, 255), "mouse": (255, 200, 0), "remote": (210, 90, 220)}
    all_summaries, raw_paths, selected_frames = [], [], {}
    row_fields = ["frame", "time_seconds", "class", "track_id", "confidence", "x1", "y1", "x2", "y2"]

    for key, title, checkpoint in models:
        model = YOLO(str(checkpoint))
        training_args = (model.ckpt or {}).get("train_args") or getattr(model.model, "args", {})
        print(f"Loading {key}: classes={model.names}; epochs={training_args.get('epochs')}; data={training_args.get('data')}", flush=True)
        source = cv2.VideoCapture(str(options.video))
        raw_path = options.out / f"{key}_raw.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {raw_path}")
        raw_paths.append(raw_path)
        ids, class_frames, object_counts = defaultdict(set), Counter(), Counter()
        frame_records, no_detection, inference_seconds = [], 0, 0.0
        trails = defaultdict(lambda: deque(maxlen=24))
        selected_frames[key] = {}
        index = 0
        start = time.perf_counter()
        try:
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                inference_start = time.perf_counter()
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", device="cpu",
                                     imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
                inference_seconds += time.perf_counter() - inference_start
                annotated = frame.copy()
                visible_classes = set()
                if len(result.boxes) == 0:
                    no_detection += 1
                for box in result.boxes:
                    name = result.names[int(box.cls.item())]
                    confidence = float(box.conf.item())
                    track_id = int(box.id.item()) if box.id is not None else None
                    xyxy = box.xyxy[0].cpu().tolist()
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    visible_classes.add(name)
                    object_counts[name] += 1
                    color = colors.get(name, (100, 255, 100))
                    if track_id is not None:
                        ids[name].add(track_id)
                        trails[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))
                        if len(trails[track_id]) > 1:
                            cv2.polylines(annotated, [np.array(trails[track_id], np.int32)], False, color, 2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    text = f"{name} #{track_id if track_id is not None else '?'} {confidence:.2f}"
                    font_scale = max(0.6, min(width, height) / 1000)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    top = max(0, y1 - th - 12)
                    cv2.rectangle(annotated, (x1, top), (min(width - 1, x1 + tw + 8), top + th + 12), color, -1)
                    cv2.putText(annotated, text, (x1 + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (15, 20, 25), 2)
                    frame_records.append({"frame": index, "time_seconds": round(index / fps, 6),
                                          "class": name, "track_id": track_id, "confidence": round(confidence, 6),
                                          **dict(zip(("x1", "y1", "x2", "y2"), xyxy))})
                class_frames.update(visible_classes)
                writer.write(annotated)
                if index in sample_indices:
                    selected_frames[key][index] = annotated.copy()
                index += 1
                if index % 60 == 0:
                    print(f"{key}: {index}/{frame_count} frames", flush=True)
        finally:
            source.release()
            writer.release()
        with (options.out / f"{key}_tracks.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=row_fields)
            csv_writer.writeheader()
            csv_writer.writerows(frame_records)
        summary = {"key": key, "label": title, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": hash_file(checkpoint), "checkpoint_training_epochs": training_args.get("epochs"),
                   "checkpoint_training_data": training_args.get("data"), "classes": model.names,
                   "frames_processed": index, "duration_seconds": index / fps,
                   "frames_without_any_detection": no_detection,
                   "frames_with_class": dict(class_frames), "box_observations": dict(object_counts),
                   "unique_track_ids_by_class": {name: sorted(values) for name, values in ids.items()},
                   "processing_wall_seconds": time.perf_counter() - start,
                   "tracking_calls_seconds_including_first_call": inference_seconds,
                   "note": "Coverage/confidence/ID counts are descriptive only, not precision, recall, mAP or ID-switch accuracy without ground truth."}
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    # Match the panel geometry, source frames and playback FPS exactly.
    scale = min(960 / width, 960 / height, 1.0)
    panel_w, panel_h = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    header = 90
    comparison_raw = options.out / "comparison_raw.mp4"
    writer = cv2.VideoWriter(str(comparison_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h + header))
    caps = [cv2.VideoCapture(str(path)) for path in raw_paths]
    comparison_samples = []
    count = 0
    try:
        while True:
            decoded = [capture.read() for capture in caps]
            if not all(ok for ok, _ in decoded):
                break
            canvas = np.full((panel_h + header, panel_w * 2, 3), (30, 24, 18), dtype=np.uint8)
            for column, (_, frame) in enumerate(decoded):
                x = column * panel_w
                canvas[header:, x:x + panel_w] = cv2.resize(frame, (panel_w, panel_h))
                title = models[column][1]
                label_scale = max(0.4, min(0.7, panel_w / 900))
                cv2.putText(canvas, title, (x + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 2)
                cv2.putText(canvas, f"CPU | conf0.25 | ByteTrack | source {count / fps:.2f}s",
                            (x + 14, 62), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, label_scale * 0.7), (190, 205, 220), 1)
            writer.write(canvas)
            if count in sample_indices:
                comparison_samples.append(canvas.copy())
            count += 1
    finally:
        writer.release()
        for capture in caps:
            capture.release()
    if count != frame_count or any(s["frames_processed"] != count for s in all_summaries):
        raise RuntimeError("A video frame count mismatch was detected")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for (key, _, _), raw_path in zip(models, raw_paths):
        encode(ffmpeg, raw_path, options.out / f"{key}_full.mp4", fps=fps)
        encode(ffmpeg, raw_path, options.out / f"{key}_{options.playback_speed:g}x.mp4",
               speed=options.playback_speed, fps=fps)
    encode(ffmpeg, comparison_raw, options.out / "comparison_full.mp4", fps=fps)
    encode(ffmpeg, comparison_raw, options.out / f"comparison_{options.playback_speed:g}x.mp4",
           speed=options.playback_speed, fps=fps)

    if comparison_samples:
        sample_w = 960
        sample_h = int(comparison_samples[0].shape[0] * sample_w / comparison_samples[0].shape[1])
        grid = Image.new("RGB", (sample_w * 2, sample_h * 4), "#101820")
        for i, sample in enumerate(comparison_samples):
            thumbnail = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)).resize((sample_w, sample_h))
            grid.paste(thumbnail, ((i % 2) * sample_w, (i // 2) * sample_h))
        grid.save(options.out / "comparison_contact_sheet.jpg", quality=92)

    metadata["models"] = all_summaries
    metadata["comparison_frames"] = count
    metadata["comparison_duration_seconds"] = count / fps
    metadata["ground_truth_available"] = False
    metadata["playback"] = {"speed": options.playback_speed, "trimmed": False,
                            "output_fps": fps * options.playback_speed,
                            "duration_seconds": count / fps / options.playback_speed,
                            "audio": False,
                            "note": "Tracking ran on every original frame before playback acceleration."}
    (options.out / "comparison_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Comparison completed: " + str(options.out), flush=True)


if __name__ == "__main__":
    main()
