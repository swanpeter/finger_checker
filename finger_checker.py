import base64
import hashlib
import io
import json
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw
from basic_setting import BasicSetting
from streamlit.errors import StreamlitSecretNotFoundError

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


DEFAULT_ANALYSIS_MODEL = "gemini-3-pro-preview"
DEFAULT_EDIT_MODEL = "gemini-3-pro-image-preview"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
OUTPUT_LONG_EDGE_BY_LABEL = {
    "2K (長辺2048px)": 2048,
    "4K (長辺4096px)": 4096,
}
AUTH = BasicSetting(default_username="", default_password="")


def get_default_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def convert_generation_config(config: dict[str, Any], style: str) -> dict[str, Any]:
    if style == "camel":
        return config

    key_map = {
        "responseMimeType": "response_mime_type",
        "responseModalities": "response_modalities",
    }
    converted: dict[str, Any] = {}
    for key, value in config.items():
        converted[key_map.get(key, key)] = value
    return converted


def build_multimodal_payload(
    prompt: str,
    image_b64: str,
    mime_type: str,
    generation_config: dict[str, Any],
    style: str,
) -> dict[str, Any]:
    if style == "camel":
        image_part = {"inlineData": {"mimeType": mime_type, "data": image_b64}}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}, image_part]}],
            "generationConfig": convert_generation_config(generation_config, style),
        }
        return payload

    image_part = {"inline_data": {"mime_type": mime_type, "data": image_b64}}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, image_part]}],
        "generation_config": convert_generation_config(generation_config, style),
    }
    return payload


def post_gemini_request(
    api_key: str,
    model_name: str,
    prompt: str,
    image_bytes: bytes,
    mime_type: str,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    normalized_model = normalize_model_name(model_name)
    url = f"{API_BASE_URL}/{normalized_model}:generateContent?key={api_key}"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    errors: list[str] = []

    for style in ("camel", "snake"):
        payload = build_multimodal_payload(
            prompt=prompt,
            image_b64=image_b64,
            mime_type=mime_type,
            generation_config=generation_config,
            style=style,
        )
        response = requests.post(url, json=payload, timeout=120)
        if response.ok:
            return response.json()
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        errors.append(f"[{style}] HTTP {response.status_code}: {detail}")

    raise RuntimeError("\n".join(errors))


def extract_text(response_json: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in response_json.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_image_part(response_json: dict[str, Any]) -> tuple[bytes, str | None] | None:
    for candidate in response_json.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue
            encoded = inline_data.get("data")
            if not encoded:
                continue
            try:
                decoded = base64.b64decode(encoded)
            except Exception:
                continue
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type")
            return decoded, mime_type
    return None


def parse_json_from_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Geminiの応答がJSONオブジェクトではありません。")
    return parsed


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def anomaly_rule(
    finger_count: int | None,
    visibility: str,
    full_hand_visible: bool,
    uncertain: bool,
) -> bool:
    # 疑わしきは罰する: 「5本が明確に見えている」以外はすべて異常候補
    if finger_count is None:
        return True
    if uncertain:
        return True
    if not (full_hand_visible or visibility == "full"):
        return True
    return finger_count != 5


def is_hand_anomalous(hand: dict[str, Any]) -> bool:
    flagged = to_bool(hand.get("is_anomalous"))
    if flagged is True:
        return True

    visibility = str(hand.get("visibility", "")).lower()
    finger_count = to_int(hand.get("finger_count"))
    full_hand_visible = to_bool(hand.get("full_hand_visible"))
    if full_hand_visible is None:
        full_hand_visible = visibility == "full"
    uncertain = to_bool(hand.get("uncertain"))
    if uncertain is None:
        uncertain = False

    return anomaly_rule(
        finger_count=finger_count,
        visibility=visibility,
        full_hand_visible=full_hand_visible,
        uncertain=uncertain,
    )


def collect_anomaly_marks(analysis_json: dict[str, Any]) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    raw_marks = analysis_json.get("anomaly_marks", [])
    if isinstance(raw_marks, list):
        for item in raw_marks:
            if not isinstance(item, dict):
                continue
            x = to_float(item.get("x"))
            y = to_float(item.get("y"))
            if x is None or y is None:
                continue
            radius = to_float(item.get("radius"))
            if 0.0 <= x <= 1.2 and 0.0 <= y <= 1.2:
                norm_x = x
                norm_y = y
            elif 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0:
                norm_x = x / 100.0
                norm_y = y / 100.0
            else:
                # Invalid coordinate format; skip and fallback to hand bbox if needed.
                continue

            if radius is None:
                norm_radius = 0.045
            elif 0.0 <= radius <= 1.0:
                norm_radius = radius
            elif 0.0 < radius <= 100.0:
                norm_radius = radius / 100.0
            else:
                norm_radius = 0.045

            marks.append(
                {
                    "x": clamp(norm_x, 0.0, 1.0),
                    "y": clamp(norm_y, 0.0, 1.0),
                    "radius": clamp(norm_radius, 0.02, 0.20),
                    "reason": str(item.get("reason", "")),
                    "label": str(item.get("finger_label", "unknown")),
                }
            )

    if marks:
        return marks

    hands = analysis_json.get("hands", [])
    if not isinstance(hands, list):
        return marks

    # Fallback: if anomaly_marks is missing, infer mark centers from anomalous hand bboxes.
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        if not is_hand_anomalous(hand):
            continue
        bbox = hand.get("bbox")
        if not isinstance(bbox, dict):
            continue
        x = to_float(bbox.get("x"))
        y = to_float(bbox.get("y"))
        w = to_float(bbox.get("width"))
        h = to_float(bbox.get("height"))
        if x is None or y is None or w is None or h is None:
            continue
        marks.append(
            {
                "x": clamp(x + (w / 2.0), 0.0, 1.0),
                "y": clamp(y + (h / 2.0), 0.0, 1.0),
                "radius": clamp(max(w, h) / 2.0, 0.02, 0.20),
                "reason": str(hand.get("reason", "")),
                "label": str(hand.get("hand_label", "unknown")),
            }
        )
    return marks


def get_image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as source:
        width, height = source.size
    return width, height


def normalize_box(x: float, y: float, w: float, h: float, min_size: float = 0.02) -> dict[str, float]:
    x = clamp(x, 0.0, 1.0 - min_size)
    y = clamp(y, 0.0, 1.0 - min_size)
    w = clamp(w, min_size, 1.0 - x)
    h = clamp(h, min_size, 1.0 - y)
    return {"x": x, "y": y, "width": w, "height": h}


def normalized_box_to_pixel(
    box: dict[str, Any], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x = to_float(box.get("x"))
    y = to_float(box.get("y"))
    w = to_float(box.get("width"))
    h = to_float(box.get("height"))
    if x is None or y is None or w is None or h is None:
        return 0, 0, 1, 1

    norm = normalize_box(x, y, w, h)
    left = int(norm["x"] * image_width)
    top = int(norm["y"] * image_height)
    right = int((norm["x"] + norm["width"]) * image_width)
    bottom = int((norm["y"] + norm["height"]) * image_height)
    width = max(1, right - left)
    height = max(1, bottom - top)
    return left, top, width, height


def pixel_box_to_normalized(
    left: int, top: int, width: int, height: int, image_width: int, image_height: int
) -> dict[str, float]:
    x = left / max(1, image_width)
    y = top / max(1, image_height)
    w = width / max(1, image_width)
    h = height / max(1, image_height)
    return normalize_box(x, y, w, h)


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1 = to_float(a.get("x")) or 0.0
    ay1 = to_float(a.get("y")) or 0.0
    ax2 = ax1 + (to_float(a.get("width")) or 0.0)
    ay2 = ay1 + (to_float(a.get("height")) or 0.0)

    bx1 = to_float(b.get("x")) or 0.0
    by1 = to_float(b.get("y")) or 0.0
    bx2 = bx1 + (to_float(b.get("width")) or 0.0)
    by2 = by1 + (to_float(b.get("height")) or 0.0)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def merge_overlapping_boxes(
    boxes: list[dict[str, Any]], iou_threshold: float = 0.9
) -> list[dict[str, Any]]:
    if not boxes:
        return []

    def area(box: dict[str, Any]) -> float:
        return (to_float(box.get("width")) or 0.0) * (to_float(box.get("height")) or 0.0)

    remaining = sorted(boxes, key=area, reverse=True)
    merged: list[dict[str, Any]] = []

    for box in remaining:
        current = dict(box)
        found_overlap = False
        for i, existing in enumerate(merged):
            if box_iou(current, existing) < iou_threshold:
                continue

            ex = to_float(existing.get("x")) or 0.0
            ey = to_float(existing.get("y")) or 0.0
            ew = to_float(existing.get("width")) or 0.0
            eh = to_float(existing.get("height")) or 0.0

            cx = to_float(current.get("x")) or 0.0
            cy = to_float(current.get("y")) or 0.0
            cw = to_float(current.get("width")) or 0.0
            ch = to_float(current.get("height")) or 0.0

            left = min(ex, cx)
            top = min(ey, cy)
            right = max(ex + ew, cx + cw)
            bottom = max(ey + eh, cy + ch)
            normalized = normalize_box(left, top, right - left, bottom - top)
            merged[i] = {
                **existing,
                **normalized,
                "source": "merged",
                "reason": str(existing.get("reason", "")) or str(current.get("reason", "")),
            }
            found_overlap = True
            break

        if not found_overlap:
            merged.append(current)
    return merged


def dedupe_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for box in boxes:
        x = to_float(box.get("x"))
        y = to_float(box.get("y"))
        w = to_float(box.get("width"))
        h = to_float(box.get("height"))
        if x is None or y is None or w is None or h is None:
            continue
        norm = normalize_box(x, y, w, h)
        key = (
            round(norm["x"], 3),
            round(norm["y"], 3),
            round(norm["width"], 3),
            round(norm["height"], 3),
        )
        if key not in unique:
            unique[key] = {**box, **norm}
    return list(unique.values())


def refine_boxes_with_opencv(
    image_bytes: bytes, boxes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    if not boxes or cv2 is None or np is None:
        return boxes, False

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return boxes, False

    image_height, image_width = image.shape[:2]
    refined_boxes: list[dict[str, Any]] = []

    for box in boxes:
        left, top, width, height = normalized_box_to_pixel(box, image_width, image_height)
        roi = image[top : top + height, left : left + width]
        if roi.size == 0:
            refined_boxes.append(box)
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 20, 40], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 20, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            refined_boxes.append(box)
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        if contour_area < float(width * height) * 0.03:
            refined_boxes.append(box)
            continue

        rx, ry, rw, rh = cv2.boundingRect(contour)
        pad_x = max(4, int(rw * 0.2))
        pad_y = max(4, int(rh * 0.2))
        global_left = max(0, left + rx - pad_x)
        global_top = max(0, top + ry - pad_y)
        global_right = min(image_width, left + rx + rw + pad_x)
        global_bottom = min(image_height, top + ry + rh + pad_y)

        normalized = pixel_box_to_normalized(
            global_left,
            global_top,
            max(1, global_right - global_left),
            max(1, global_bottom - global_top),
            image_width,
            image_height,
        )
        refined_boxes.append(
            {
                **box,
                **normalized,
                "source": "opencv_refined",
            }
        )

    return refined_boxes, True


def collect_fix_boxes(
    analysis_json: dict[str, Any], image_bytes: bytes, use_opencv_refine: bool
) -> tuple[list[dict[str, Any]], bool]:
    boxes: list[dict[str, Any]] = []

    for mark in collect_anomaly_marks(analysis_json):
        x = to_float(mark.get("x"))
        y = to_float(mark.get("y"))
        radius = to_float(mark.get("radius"))
        if x is None or y is None:
            continue
        radius = radius if radius is not None else 0.045
        radius = clamp(radius * 1.8, 0.03, 0.28)
        normalized = normalize_box(x - radius, y - radius, radius * 2.0, radius * 2.0)
        boxes.append(
            {
                **normalized,
                "source": "anomaly_mark",
                "label": str(mark.get("label", "unknown")),
                "reason": str(mark.get("reason", "")),
            }
        )

    hands = analysis_json.get("hands", [])
    if isinstance(hands, list):
        for hand in hands:
            if not isinstance(hand, dict) or not is_hand_anomalous(hand):
                continue
            bbox = hand.get("bbox")
            if not isinstance(bbox, dict):
                continue
            x = to_float(bbox.get("x"))
            y = to_float(bbox.get("y"))
            w = to_float(bbox.get("width"))
            h = to_float(bbox.get("height"))
            if x is None or y is None or w is None or h is None:
                continue
            pad_w = w * 0.25
            pad_h = h * 0.25
            normalized = normalize_box(x - pad_w, y - pad_h, w + (pad_w * 2), h + (pad_h * 2))
            boxes.append(
                {
                    **normalized,
                    "source": "hand_bbox",
                    "label": str(hand.get("hand_label", "unknown")),
                    "reason": str(hand.get("reason", "")),
                }
            )

    boxes = merge_overlapping_boxes(dedupe_boxes(boxes), iou_threshold=0.92)

    opencv_used = False
    if use_opencv_refine and boxes:
        boxes, opencv_used = refine_boxes_with_opencv(image_bytes, boxes)
        boxes = merge_overlapping_boxes(dedupe_boxes(boxes), iou_threshold=0.92)

    return boxes, opencv_used


def build_box_preview_image(image_bytes: bytes, boxes: list[dict[str, Any]]) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    line_width = max(3, int(min(width, height) * 0.004))

    for index, box in enumerate(boxes, start=1):
        left, top, w, h = normalized_box_to_pixel(box, width, height)
        right = left + w
        bottom = top + h
        draw.rectangle((left, top, right, bottom), outline=(255, 140, 0, 255), width=line_width)

        tag_w = 28
        tag_h = 18
        tag_x = clamp(float(left + 2), 0.0, float(max(0, width - tag_w)))
        tag_y = clamp(float(top - tag_h - 4), 0.0, float(max(0, height - tag_h)))
        draw.rounded_rectangle(
            (int(tag_x), int(tag_y), int(tag_x) + tag_w, int(tag_y) + tag_h),
            radius=4,
            fill=(255, 140, 0, 190),
        )
        draw.text((int(tag_x) + 7, int(tag_y) + 2), str(index), fill=(255, 255, 255, 255))

    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def format_fix_boxes_for_prompt(
    fix_boxes: list[dict[str, Any]], image_width: int, image_height: int
) -> str:
    if not fix_boxes:
        return "修正対象の参照領域は指定なし。"

    lines: list[str] = []
    for index, box in enumerate(fix_boxes, start=1):
        left, top, width, height = normalized_box_to_pixel(box, image_width, image_height)
        right = left + width
        bottom = top + height
        source = str(box.get("source", "unknown"))
        reason = str(box.get("reason", ""))
        lines.append(
            f"- R{index}: left={left}, top={top}, right={right}, bottom={bottom}, "
            f"source={source}, reason={reason}"
        )
    return "\n".join(lines)


def build_marked_image(image_bytes: bytes, anomaly_marks: list[dict[str, Any]]) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    line_width = max(3, int(min(width, height) * 0.005))

    for index, mark in enumerate(anomaly_marks, start=1):
        center_x = int(mark["x"] * width)
        center_y = int(mark["y"] * height)
        radius_px = max(12, int(mark["radius"] * min(width, height)))
        left = center_x - radius_px
        top = center_y - radius_px
        right = center_x + radius_px
        bottom = center_y + radius_px

        draw.ellipse((left, top, right, bottom), outline=(255, 0, 0, 255), width=line_width)
        draw.ellipse(
            (center_x - line_width, center_y - line_width, center_x + line_width, center_y + line_width),
            fill=(255, 0, 0, 255),
        )

        tag_w = 24
        tag_h = 18
        tag_x = clamp(float(center_x + radius_px + 6), 0.0, float(max(0, width - tag_w)))
        tag_y = clamp(float(center_y - radius_px - tag_h - 4), 0.0, float(max(0, height - tag_h)))
        draw.rounded_rectangle(
            (int(tag_x), int(tag_y), int(tag_x) + tag_w, int(tag_y) + tag_h),
            radius=4,
            fill=(255, 0, 0, 190),
        )
        draw.text((int(tag_x) + 7, int(tag_y) + 2), str(index), fill=(255, 255, 255, 255))

    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def normalize_hand_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"left", "right", "unknown"}:
        return label
    return "unknown"


def normalize_visibility(value: Any) -> str:
    visibility = str(value or "").strip().lower()
    if visibility in {"full", "partial", "unclear"}:
        return visibility
    return "unclear"


def normalize_bbox_values(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, float] | None:
    if w <= 0 or h <= 0:
        return None

    # Already normalized [0, 1]
    if max(x, y, w, h) <= 1.2:
        return normalize_box(x, y, w, h, min_size=0.01)

    # Percentage style [0, 100]
    if max(x, y, w, h) <= 100.0:
        return normalize_box(x / 100.0, y / 100.0, w / 100.0, h / 100.0, min_size=0.01)

    # Pixel style, convert using image size
    if image_width and image_height and image_width > 0 and image_height > 0:
        if x <= image_width * 1.05 and y <= image_height * 1.05 and w <= image_width * 1.2 and h <= image_height * 1.2:
            return normalize_box(
                x / float(image_width),
                y / float(image_height),
                w / float(image_width),
                h / float(image_height),
                min_size=0.01,
            )

    return None


def normalize_hand_bbox(
    raw_bbox: Any,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, float] | None:
    if not isinstance(raw_bbox, dict):
        return None
    x = to_float(raw_bbox.get("x"))
    y = to_float(raw_bbox.get("y"))
    w = to_float(raw_bbox.get("width"))
    h = to_float(raw_bbox.get("height"))
    if x is None or y is None or w is None or h is None:
        return None
    return normalize_bbox_values(
        x=x,
        y=y,
        w=w,
        h=h,
        image_width=image_width,
        image_height=image_height,
    )


def normalize_detected_hands(
    raw_hands: Any,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_hands, list):
        return normalized

    for index, hand in enumerate(raw_hands, start=1):
        if not isinstance(hand, dict):
            continue
        bbox = normalize_hand_bbox(
            hand.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue
        normalized.append(
            {
                "hand_id": str(hand.get("hand_id", f"H{index}")),
                "person_id": str(hand.get("person_id", f"P{index}")),
                "hand_label": normalize_hand_label(hand.get("hand_label")),
                "visibility": normalize_visibility(hand.get("visibility")),
                "reason": str(hand.get("reason", "")),
                "bbox": bbox,
            }
        )
    return normalized


def crop_image_by_normalized_box(
    image_bytes: bytes, bbox: dict[str, Any], margin_ratio: float = 0.25
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")

    width, height = image.size
    left, top, box_w, box_h = normalized_box_to_pixel(bbox, width, height)
    pad_w = int(box_w * margin_ratio)
    pad_h = int(box_h * margin_ratio)
    crop_left = max(0, left - pad_w)
    crop_top = max(0, top - pad_h)
    crop_right = min(width, left + box_w + pad_w)
    crop_bottom = min(height, top + box_h + pad_h)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        crop_left, crop_top, crop_right, crop_bottom = 0, 0, width, height

    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    output = io.BytesIO()
    crop.save(output, format="PNG")
    return output.getvalue()


def detect_hands_strict(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_width, image_height = get_image_size(image_bytes)
    prompt = """
あなたは厳密な手検出器です。画像内の人間の手を漏れなく列挙してください。
出力はJSONのみ。説明文やMarkdownは禁止。

重要ルール:
- アニメ/3D/デフォルメでも例外を作らない。
- 「本来は5本だから」などの補正をしない。
- 実際に画像に描かれている手を数える。
- 部分的に見える手も含める。
- bboxは画像全体に対する正規化座標(0.0-1.0)。

必須JSON:
{
  "people_count": int,
  "hands": [
    {
      "hand_id": "H1",
      "person_id": "P1",
      "hand_label": "left|right|unknown",
      "visibility": "full|partial|unclear",
      "bbox": {
        "x": float,
        "y": float,
        "width": float,
        "height": float
      },
      "reason": "日本語の短い理由"
    }
  ],
  "summary": "日本語で1文"
}
""".strip()

    generation_config = {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    raw_response = post_gemini_request(
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
        generation_config=generation_config,
    )
    raw_text = extract_text(raw_response)
    if not raw_text:
        raise ValueError("手検出レスポンスにテキストが含まれていません。")
    parsed = parse_json_from_text(raw_text)

    people_count = to_int(parsed.get("people_count"))
    people_count = max(0, people_count if people_count is not None else 0)
    hands = normalize_detected_hands(
        parsed.get("hands"),
        image_width=image_width,
        image_height=image_height,
    )
    if people_count == 0 and hands:
        person_ids = {h["person_id"] for h in hands if h.get("person_id")}
        if person_ids:
            people_count = len(person_ids)

    result = {
        "people_count": people_count,
        "hands": hands,
        "summary": str(parsed.get("summary", "")),
    }
    return result, raw_response


def count_fingers_strict_for_hand(
    api_key: str,
    model_name: str,
    hand_crop_bytes: bytes,
    hand_label: str,
    visibility: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = f"""
あなたは厳密な指カウンタです。入力画像は1つの手を中心に切り出したものです。
出力はJSONのみ。説明文やMarkdownは禁止。

入力コンテキスト:
- hand_label: {hand_label}
- visibility_hint: {visibility}

超厳密ルール:
- 「実際に描かれている指の本数」をそのまま数える。
- アニメ/3Dの簡略化を正当化しない。
- 「人間だから5本のはず」と推定しない。
- 見切れ・重なりで不明な指はカウントしない。
- 4本に見えるなら4、5本に見えるなら5を返す。

必須JSON:
{{
  "visible_finger_count": int,
  "full_hand_visible": bool,
  "uncertain": bool,
  "confidence": float,
  "reason": "日本語の短い理由"
}}
""".strip()

    generation_config = {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    raw_response = post_gemini_request(
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        image_bytes=hand_crop_bytes,
        mime_type="image/png",
        generation_config=generation_config,
    )
    raw_text = extract_text(raw_response)
    if not raw_text:
        raise ValueError("手の指カウントレスポンスにテキストが含まれていません。")
    parsed = parse_json_from_text(raw_text)

    finger_count = to_int(parsed.get("visible_finger_count"))
    if finger_count is None:
        finger_count = to_int(parsed.get("finger_count"))
    finger_count = 0 if finger_count is None else max(0, min(10, finger_count))

    full_hand_visible = to_bool(parsed.get("full_hand_visible"))
    if full_hand_visible is None:
        full_hand_visible = visibility == "full"

    uncertain = to_bool(parsed.get("uncertain"))
    if uncertain is None:
        uncertain = False

    confidence = to_float(parsed.get("confidence"))
    confidence = 0.0 if confidence is None else clamp(confidence, 0.0, 1.0)

    result = {
        "visible_finger_count": finger_count,
        "full_hand_visible": full_hand_visible,
        "uncertain": uncertain,
        "confidence": confidence,
        "reason": str(parsed.get("reason", "")),
    }
    return result, raw_response


def build_anomaly_marks_from_hands(hands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marks: list[dict[str, Any]] = []
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        if not is_hand_anomalous(hand):
            continue
        bbox = hand.get("bbox")
        if not isinstance(bbox, dict):
            continue
        x = to_float(bbox.get("x"))
        y = to_float(bbox.get("y"))
        w = to_float(bbox.get("width"))
        h = to_float(bbox.get("height"))
        if x is None or y is None or w is None or h is None:
            continue

        marks.append(
            {
                "person_id": str(hand.get("person_id", "unknown")),
                "hand_label": str(hand.get("hand_label", "unknown")),
                "finger_label": "unknown",
                "x": clamp(x + (w / 2.0), 0.0, 1.0),
                "y": clamp(y + (h / 2.0), 0.0, 1.0),
                "radius": clamp(max(w, h) / 2.0, 0.03, 0.25),
                "reason": str(hand.get("reason", "")),
            }
        )
    return marks


def build_analysis_result_from_hands(
    people_count: int,
    hands: list[dict[str, Any]],
    summary: str | None = None,
    detection_summary: str = "",
    anomaly_marks_hint: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_people_count = max(0, people_count)
    if normalized_people_count == 0 and hands:
        person_ids = {str(h.get("person_id", "")) for h in hands if str(h.get("person_id", ""))}
        if person_ids:
            normalized_people_count = len(person_ids)

    hands_count = len(hands)
    fingers_count = sum(to_int(hand.get("finger_count")) or 0 for hand in hands)
    confidence_values = [clamp(to_float(h.get("confidence")) or 0.0, 0.0, 1.0) for h in hands]
    confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    )

    analysis_source = {
        "hands": hands,
        "anomaly_marks": anomaly_marks_hint if isinstance(anomaly_marks_hint, list) else [],
    }
    anomaly_marks = collect_anomaly_marks(analysis_source)
    anomaly_count = len(anomaly_marks)

    if summary is None or not str(summary).strip():
        summary = (
            f"人数{normalized_people_count}人、手{hands_count}本、可視の指は合計{fingers_count}本。"
            f"5本未確認または不確実な候補は{anomaly_count}件です。"
        )

    return {
        "people_count": normalized_people_count,
        "hands_count": hands_count,
        "fingers_count": fingers_count,
        "hands": hands,
        "anomaly_marks": anomaly_marks,
        "confidence": round(confidence, 3),
        "summary": str(summary),
        "detection_summary": detection_summary,
    }


def normalize_hands_from_batch(
    raw_hands: Any,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
    if not isinstance(raw_hands, list):
        return hands

    for index, raw in enumerate(raw_hands, start=1):
        if not isinstance(raw, dict):
            continue
        bbox = normalize_hand_bbox(
            raw.get("bbox"),
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            continue

        visibility = normalize_visibility(raw.get("visibility"))
        full_hand_visible = to_bool(raw.get("full_hand_visible"))
        if full_hand_visible is None:
            full_hand_visible = visibility == "full"

        uncertain = to_bool(raw.get("uncertain"))
        if uncertain is None:
            uncertain = False

        finger_count = to_int(raw.get("finger_count"))
        if finger_count is None:
            finger_count = to_int(raw.get("visible_finger_count"))
        if finger_count is None:
            finger_count = 0
        finger_count = max(0, min(10, finger_count))

        confidence = to_float(raw.get("confidence"))
        confidence = clamp(confidence if confidence is not None else 0.0, 0.0, 1.0)

        hands.append(
            {
                "hand_id": str(raw.get("hand_id", f"H{index}")),
                "person_id": str(raw.get("person_id", f"P{index}")),
                "hand_label": normalize_hand_label(raw.get("hand_label")),
                "finger_count": finger_count,
                "visibility": visibility,
                "full_hand_visible": full_hand_visible,
                "uncertain": uncertain,
                "is_anomalous": anomaly_rule(
                    finger_count=finger_count,
                    visibility=visibility,
                    full_hand_visible=full_hand_visible,
                    uncertain=uncertain,
                ),
                "reason": str(raw.get("reason", "")),
                "bbox": bbox,
                "confidence": confidence,
            }
        )
    return hands


def analyze_image_batch(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_width, image_height = get_image_size(image_bytes)
    prompt = """
あなたは厳密な画像QA検査員です。入力画像を一括で解析してください。
出力はJSONのみ。説明文やMarkdownは禁止。

超厳密ルール:
- 画像に描かれている見た目だけを根拠に判定。
- 「人間だから5本のはず」の推定は禁止。
- アニメ/3D/デフォルメでも例外を作らない。
- 見切れ・重なりで断定できない場合は uncertain=true。

必須JSON:
{
  "people_count": int,
  "hands": [
    {
      "hand_id": "H1",
      "person_id": "P1",
      "hand_label": "left|right|unknown",
      "visible_finger_count": int,
      "visibility": "full|partial|unclear",
      "full_hand_visible": bool,
      "uncertain": bool,
      "confidence": float,
      "reason": "日本語の短い理由",
      "bbox": {
        "x": float,
        "y": float,
        "width": float,
        "height": float
      }
    }
  ],
  "anomaly_marks": [
    {
      "person_id": "P1",
      "hand_label": "left|right|unknown",
      "finger_label": "thumb|index|middle|ring|pinky|unknown",
      "x": float,
      "y": float,
      "radius": float,
      "reason": "日本語の短い理由"
    }
  ],
  "summary": "日本語で1文"
}
""".strip()

    generation_config = {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    raw_response = post_gemini_request(
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
        generation_config=generation_config,
    )
    raw_text = extract_text(raw_response)
    if not raw_text:
        raise ValueError("一括解析レスポンスにテキストが含まれていません。")
    parsed = parse_json_from_text(raw_text)

    people_count = to_int(parsed.get("people_count"))
    if people_count is None:
        people_count = 0
    hands = normalize_hands_from_batch(
        parsed.get("hands"),
        image_width=image_width,
        image_height=image_height,
    )
    result = build_analysis_result_from_hands(
        people_count=people_count,
        hands=hands,
        summary=str(parsed.get("summary", "")).strip() or None,
        detection_summary="一括解析モード",
        anomaly_marks_hint=parsed.get("anomaly_marks") if isinstance(parsed.get("anomaly_marks"), list) else [],
    )
    return result, raw_response


def analyze_image_per_hand(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hand_detection, hand_detection_raw = detect_hands_strict(
        api_key=api_key,
        model_name=model_name,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )

    detected_hands = hand_detection.get("hands", [])
    if not isinstance(detected_hands, list):
        detected_hands = []

    hands: list[dict[str, Any]] = []
    per_hand_raw: list[dict[str, Any]] = []

    for index, detected in enumerate(detected_hands, start=1):
        if not isinstance(detected, dict):
            continue
        bbox = detected.get("bbox")
        if not isinstance(bbox, dict):
            continue

        crop_bytes = crop_image_by_normalized_box(image_bytes, bbox, margin_ratio=0.25)
        hand_id = str(detected.get("hand_id", f"H{index}"))
        hand_label = normalize_hand_label(detected.get("hand_label"))
        visibility = normalize_visibility(detected.get("visibility"))

        try:
            finger_result, finger_raw = count_fingers_strict_for_hand(
                api_key=api_key,
                model_name=model_name,
                hand_crop_bytes=crop_bytes,
                hand_label=hand_label,
                visibility=visibility,
            )
        except Exception as error:
            finger_result = {
                "visible_finger_count": 0,
                "full_hand_visible": visibility == "full",
                "uncertain": True,
                "confidence": 0.0,
                "reason": f"手ごとの再判定に失敗: {error}",
            }
            finger_raw = {"error": str(error)}

        finger_count = max(0, int(finger_result.get("visible_finger_count", 0)))
        full_hand_visible = bool(finger_result.get("full_hand_visible", visibility == "full"))
        confidence_value = to_float(finger_result.get("confidence"))
        confidence = clamp(confidence_value if confidence_value is not None else 0.0, 0.0, 1.0)
        uncertain = bool(finger_result.get("uncertain", False))

        hand_item = {
            "hand_id": hand_id,
            "person_id": str(detected.get("person_id", f"P{index}")),
            "hand_label": hand_label,
            "finger_count": finger_count,
            "visibility": visibility,
            "is_anomalous": anomaly_rule(
                finger_count=finger_count,
                visibility=visibility,
                full_hand_visible=full_hand_visible,
                uncertain=uncertain,
            ),
            "reason": str(finger_result.get("reason") or detected.get("reason", "")),
            "bbox": bbox,
            "full_hand_visible": full_hand_visible,
            "uncertain": uncertain,
            "confidence": confidence,
        }
        hands.append(hand_item)
        per_hand_raw.append(
            {
                "hand_id": hand_id,
                "bbox": bbox,
                "finger_result": finger_result,
                "finger_raw_response": finger_raw,
            }
        )

    people_count = to_int(hand_detection.get("people_count"))
    people_count = max(0, people_count if people_count is not None else 0)
    result = build_analysis_result_from_hands(
        people_count=people_count,
        hands=hands,
        summary=None,
        detection_summary=str(hand_detection.get("summary", "")),
        anomaly_marks_hint=None,
    )
    raw = {
        "hand_detection": hand_detection_raw,
        "per_hand_finger_checks": per_hand_raw,
    }
    return result, raw


def analyze_image(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    analysis_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if analysis_mode == "batch":
        return analyze_image_batch(
            api_key=api_key,
            model_name=model_name,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

    return analyze_image_per_hand(
        api_key=api_key,
        model_name=model_name,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )


def build_edit_prompt(
    analysis_json: dict[str, Any],
    fix_boxes: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    extra_instruction: str,
) -> str:
    fix_box_text = format_fix_boxes_for_prompt(fix_boxes, image_width, image_height)
    base_prompt = f"""
この画像を編集してください。
目的: 人間の指のみを必要最小限修正し、はっきり見える手は5本指にしてください。

厳守:
- 変更対象は手と指のみ。
- 顔、人物ID、体、服、背景、ライティング、画角、全体スタイルは維持。
- 人数は変えない。
- 新しい物体は追加しない。
- まず下記の参照領域内の指を優先して修正する。
- R1..Rn を全て対象にする。1つでも未修正を残さない。
- どれか1箇所だけ修正して終了しない。
- 参照領域外の変更は最小限。
- 参照領域は内部参照用。座標情報を見た目として描画してはいけない。
- 枠線・四角形・丸・線・矢印・文字・番号・注釈を画像上に絶対に描かない。
- 黄/オレンジ/赤の枠線を1つでも描いてはいけない。
- 最終出力は通常の完成画像のみ。デバッグ表示やガイド表示は禁止。

修正対象参照領域（ピクセル座標。描画禁止）:
{fix_box_text}

解析結果（参照用）:
{json.dumps(analysis_json, ensure_ascii=True)}
""".strip()

    if extra_instruction.strip():
        return f"{base_prompt}\n\n追加指示:\n{extra_instruction.strip()}"
    return base_prompt


def decode_image_to_bgr(image_bytes: bytes) -> Any | None:
    if cv2 is None or np is None:
        return None
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    if np_bytes.size == 0:
        return None
    return cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)


def build_overlay_artifact_mask(image_bgr: Any) -> Any | None:
    if cv2 is None or np is None or image_bgr is None:
        return None

    height, width = image_bgr.shape[:2]
    if width <= 0 or height <= 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array([12, 60, 120], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    orange_mask = cv2.inRange(
        hsv,
        np.array([0, 100, 120], dtype=np.uint8),
        np.array([15, 255, 255], dtype=np.uint8),
    )
    color_mask = cv2.bitwise_or(yellow_mask, orange_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_mask = np.zeros_like(color_mask)
    image_area = float(width * height)

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < 80.0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rect_area = float(max(1, w * h))
        if rect_area < 300.0 or rect_area > image_area * 0.2:
            continue

        aspect = w / float(max(1, h))
        if aspect < 0.2 or aspect > 5.0:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 8:
            continue

        # 枠線っぽい輪郭だけを対象にする（塗りつぶしオブジェクトは除外）。
        fill_ratio = contour_area / rect_area
        if fill_ratio > 0.45:
            continue

        stroke = max(2, int(round(min(w, h) * 0.07)))
        cv2.drawContours(result_mask, [contour], -1, 255, thickness=stroke)

    result_mask = cv2.dilate(result_mask, kernel, iterations=2)
    return result_mask


def has_overlay_artifacts(image_bytes: bytes) -> bool:
    if cv2 is None or np is None:
        return False
    image_bgr = decode_image_to_bgr(image_bytes)
    if image_bgr is None:
        return False
    mask = build_overlay_artifact_mask(image_bgr)
    if mask is None:
        return False
    return int((mask > 0).sum()) >= 120


def remove_overlay_artifacts_with_inpaint(image_bytes: bytes) -> bytes | None:
    if cv2 is None or np is None:
        return None
    image_bgr = decode_image_to_bgr(image_bytes)
    if image_bgr is None:
        return None
    mask = build_overlay_artifact_mask(image_bgr)
    if mask is None or int((mask > 0).sum()) < 120:
        return None
    inpainted = cv2.inpaint(image_bgr, mask, 4, cv2.INPAINT_TELEA)
    ok, encoded = cv2.imencode(".png", inpainted)
    if not ok:
        return None
    return bytes(encoded.tobytes())


def build_cleanup_prompt() -> str:
    return """
この画像に重ね描きされた注釈だけを除去してください。

厳守:
- 黄/オレンジ/赤の四角枠、丸、線、矢印、文字、番号、ラベルを完全に消す。
- 人物、手、指、顔、服、背景、ライティング、画角、全体スタイルは変えない。
- 人数を変えない。新規オブジェクトを追加しない。
- 最終出力は注釈のない完成画像1枚のみ。
""".strip()


def cleanup_overlay_artifacts(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> tuple[bytes, str | None, dict[str, Any]]:
    generation_config = {
        "temperature": 0.1,
        "responseModalities": ["TEXT", "IMAGE"],
    }
    raw_response = post_gemini_request(
        api_key=api_key,
        model_name=model_name,
        prompt=build_cleanup_prompt(),
        image_bytes=image_bytes,
        mime_type=mime_type,
        generation_config=generation_config,
    )
    image_part = extract_image_part(raw_response)
    if image_part is None:
        response_text = extract_text(raw_response)
        raise ValueError(
            "注釈除去レスポンスに画像データが含まれていません。"
            f"モデル出力: {response_text or '(empty)'}"
        )
    return image_part[0], image_part[1], raw_response


def image_to_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def build_local_patch_edit_prompt(
    target_left: int,
    target_top: int,
    target_right: int,
    target_bottom: int,
    crop_width: int,
    crop_height: int,
    box: dict[str, Any],
    extra_instruction: str,
) -> str:
    source = str(box.get("source", "unknown"))
    reason = str(box.get("reason", "")).strip()
    base_prompt = f"""
この画像は元画像の一部を切り出したパッチです。
目的: 指の本数だけを修正し、明確に見える手は5本指にしてください。

編集対象領域（このパッチ内のピクセル座標）:
left={target_left}, top={target_top}, right={target_right}, bottom={target_bottom}

厳守:
- 編集は編集対象領域内の手・指のみ。
- 編集対象領域の外側は変更しない。
- 顔、体、服、背景、ライティング、色調、構図、画角、スタイルを維持。
- 人物数・物体数は変更しない。
- 文字、番号、枠線、四角、丸、矢印など注釈を描かない。
- 最終出力は通常画像のみ。

補足:
- source={source}
- reason={reason or '(none)'}
- patch_size={crop_width}x{crop_height}
""".strip()

    if extra_instruction.strip():
        return f"{base_prompt}\n\n追加指示:\n{extra_instruction.strip()}"
    return base_prompt


def edit_patch_with_nanobanana(
    api_key: str,
    model_name: str,
    patch_bytes: bytes,
    patch_mime_type: str,
    prompt: str,
) -> tuple[bytes, str | None, dict[str, Any]]:
    generation_config = {
        "temperature": 0.15,
        "responseModalities": ["TEXT", "IMAGE"],
    }
    raw_response = post_gemini_request(
        api_key=api_key,
        model_name=model_name,
        prompt=prompt,
        image_bytes=patch_bytes,
        mime_type=patch_mime_type,
        generation_config=generation_config,
    )
    image_part = extract_image_part(raw_response)
    if image_part is None:
        response_text = extract_text(raw_response)
        raise ValueError(
            "部分編集レスポンスに画像データが含まれていません。"
            f"モデル出力: {response_text or '(empty)'}"
        )

    edited_bytes = image_part[0]
    edited_mime = image_part[1] or "image/png"
    debug_info: dict[str, Any] = {
        "raw": raw_response,
        "cleanup_passes": [],
        "inpaint_fallback_applied": False,
    }

    working_bytes = edited_bytes
    working_mime = edited_mime
    for _ in range(2):
        if not has_overlay_artifacts(working_bytes):
            break
        try:
            cleaned_bytes, cleaned_mime, cleanup_raw = cleanup_overlay_artifacts(
                api_key=api_key,
                model_name=model_name,
                image_bytes=working_bytes,
                mime_type=working_mime,
            )
            debug_info["cleanup_passes"].append({"ok": True, "raw": cleanup_raw})
            working_bytes = cleaned_bytes
            working_mime = cleaned_mime or "image/png"
        except Exception as error:
            debug_info["cleanup_passes"].append({"ok": False, "error": str(error)})
            break

    if has_overlay_artifacts(working_bytes):
        inpainted = remove_overlay_artifacts_with_inpaint(working_bytes)
        if inpainted:
            working_bytes = inpainted
            working_mime = "image/png"
            debug_info["inpaint_fallback_applied"] = True

    return working_bytes, working_mime, debug_info


def resize_image_to_long_edge(image_bytes: bytes, long_edge: int) -> tuple[bytes, str]:
    if long_edge <= 0:
        return image_bytes, "image/png"

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue(), "image/png"

        if width >= height:
            new_width = long_edge
            new_height = max(1, int(round(height * (long_edge / width))))
        else:
            new_height = long_edge
            new_width = max(1, int(round(width * (long_edge / height))))

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((new_width, new_height), resampling)

    output = io.BytesIO()
    resized.save(output, format="PNG")
    return output.getvalue(), "image/png"


def edit_image_with_nanobanana(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    analysis_json: dict[str, Any],
    fix_boxes: list[dict[str, Any]],
    output_long_edge: int,
    extra_instruction: str,
) -> tuple[bytes, str | None, dict[str, Any]]:
    # 厳密モード: bboxごとの局所編集結果だけを元画像に貼り戻し、bbox外の画素を固定する。
    with Image.open(io.BytesIO(image_bytes)) as source:
        base_image = source.convert("RGB")
    image_width, image_height = base_image.size

    debug_bundle: dict[str, Any] = {
        "mode": "strict_local_patch",
        "regions": [],
        "analysis_summary": str(analysis_json.get("summary", "")),
    }

    if not fix_boxes:
        resized_bytes, resized_mime = resize_image_to_long_edge(image_bytes, output_long_edge)
        debug_bundle["skipped_reason"] = "fix_boxes_empty"
        return resized_bytes, resized_mime or "image/png", debug_bundle

    sorted_boxes = sorted(
        fix_boxes,
        key=lambda box: (to_float(box.get("width")) or 0.0) * (to_float(box.get("height")) or 0.0),
        reverse=True,
    )

    working_image = base_image.copy()
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for region_index, box in enumerate(sorted_boxes, start=1):
        core_left, core_top, core_width, core_height = normalized_box_to_pixel(
            box,
            image_width,
            image_height,
        )
        core_right = min(image_width, core_left + core_width)
        core_bottom = min(image_height, core_top + core_height)
        core_width = max(1, core_right - core_left)
        core_height = max(1, core_bottom - core_top)

        if core_width <= 0 or core_height <= 0:
            debug_bundle["regions"].append(
                {
                    "index": region_index,
                    "ok": False,
                    "error": "invalid_core_box",
                }
            )
            continue

        pad_x = max(12, int(round(core_width * 0.35)))
        pad_y = max(12, int(round(core_height * 0.35)))
        context_left = max(0, core_left - pad_x)
        context_top = max(0, core_top - pad_y)
        context_right = min(image_width, core_right + pad_x)
        context_bottom = min(image_height, core_bottom + pad_y)
        context_width = max(1, context_right - context_left)
        context_height = max(1, context_bottom - context_top)

        target_left = core_left - context_left
        target_top = core_top - context_top
        target_right = target_left + core_width
        target_bottom = target_top + core_height

        context_crop = working_image.crop((context_left, context_top, context_right, context_bottom))
        context_bytes = image_to_png_bytes(context_crop)

        local_prompt = build_local_patch_edit_prompt(
            target_left=target_left,
            target_top=target_top,
            target_right=target_right,
            target_bottom=target_bottom,
            crop_width=context_width,
            crop_height=context_height,
            box=box,
            extra_instruction=extra_instruction,
        )

        try:
            edited_patch_bytes, edited_patch_mime, patch_debug = edit_patch_with_nanobanana(
                api_key=api_key,
                model_name=model_name,
                patch_bytes=context_bytes,
                patch_mime_type="image/png",
                prompt=local_prompt,
            )

            with Image.open(io.BytesIO(edited_patch_bytes)) as edited_patch_source:
                edited_patch = edited_patch_source.convert("RGB")

            if edited_patch.size != (context_width, context_height):
                edited_patch = edited_patch.resize((context_width, context_height), resampling)

            updated_core = edited_patch.crop((target_left, target_top, target_right, target_bottom))
            if updated_core.size != (core_width, core_height):
                updated_core = updated_core.resize((core_width, core_height), resampling)
            working_image.paste(updated_core, (core_left, core_top))

            debug_bundle["regions"].append(
                {
                    "index": region_index,
                    "ok": True,
                    "core_box": {
                        "left": core_left,
                        "top": core_top,
                        "right": core_right,
                        "bottom": core_bottom,
                    },
                    "context_box": {
                        "left": context_left,
                        "top": context_top,
                        "right": context_right,
                        "bottom": context_bottom,
                    },
                    "mime": edited_patch_mime or "image/png",
                    "debug": patch_debug,
                }
            )
        except Exception as error:
            debug_bundle["regions"].append(
                {
                    "index": region_index,
                    "ok": False,
                    "error": str(error),
                    "core_box": {
                        "left": core_left,
                        "top": core_top,
                        "right": core_right,
                        "bottom": core_bottom,
                    },
                }
            )

    composited_bytes = image_to_png_bytes(working_image)
    resized_bytes, resized_mime = resize_image_to_long_edge(composited_bytes, output_long_edge)
    return resized_bytes, resized_mime or "image/png", debug_bundle


def detect_anomalies(hands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for hand in hands:
        if is_hand_anomalous(hand):
            anomalies.append(hand)
    return anomalies


def ensure_file_state(uploaded_bytes: bytes) -> None:
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    if st.session_state.get("file_digest") == digest:
        return

    st.session_state["file_digest"] = digest
    st.session_state["analysis_result"] = None
    st.session_state["analysis_raw"] = None
    st.session_state["anomaly_marks"] = None
    st.session_state["marked_image"] = None
    st.session_state["fix_boxes"] = None
    st.session_state["fix_boxes_preview"] = None
    st.session_state["opencv_used"] = None
    st.session_state["edited_image"] = None
    st.session_state["edited_mime"] = None
    st.session_state["edit_raw"] = None


def run_pipeline_for_image(
    api_key: str,
    analysis_model: str,
    analysis_mode: str,
    image_bytes: bytes,
    mime_type: str,
    enable_marking: bool,
    enable_opencv_bbox: bool,
    enable_fix: bool,
    edit_model: str,
    output_long_edge: int,
    extra_instruction: str,
) -> dict[str, Any]:
    analysis_result, raw_response = analyze_image(
        api_key=api_key,
        model_name=analysis_model,
        image_bytes=image_bytes,
        mime_type=mime_type,
        analysis_mode=analysis_mode,
    )
    analysis_result["analysis_mode"] = analysis_mode

    anomaly_marks = collect_anomaly_marks(analysis_result)
    fix_boxes, opencv_used = collect_fix_boxes(
        analysis_result,
        image_bytes=image_bytes,
        use_opencv_refine=enable_opencv_bbox,
    )

    marked_image = None
    if enable_marking and anomaly_marks:
        marked_image = build_marked_image(image_bytes, anomaly_marks)

    fix_boxes_preview = None
    if fix_boxes:
        fix_boxes_preview = build_box_preview_image(image_bytes, fix_boxes)

    edited_image = None
    edited_mime = None
    edit_raw = None
    auto_fix_skipped_reason = None

    if enable_fix:
        if not fix_boxes:
            auto_fix_skipped_reason = "修正対象bboxがありません。"
        else:
            edited_image, edited_mime, edit_raw = edit_image_with_nanobanana(
                api_key=api_key,
                model_name=edit_model,
                image_bytes=image_bytes,
                mime_type=mime_type,
                analysis_json=analysis_result,
                fix_boxes=fix_boxes,
                output_long_edge=output_long_edge,
                extra_instruction=extra_instruction,
            )

    return {
        "analysis_result": analysis_result,
        "analysis_raw": raw_response,
        "anomaly_marks": anomaly_marks,
        "fix_boxes": fix_boxes,
        "opencv_used": opencv_used,
        "marked_image": marked_image,
        "fix_boxes_preview": fix_boxes_preview,
        "edited_image": edited_image,
        "edited_mime": edited_mime,
        "edit_raw": edit_raw,
        "auto_fix_skipped_reason": auto_fix_skipped_reason,
    }


def build_uploaded_files_digest(uploaded_files: list[Any]) -> str:
    hash_obj = hashlib.sha256()
    for file_obj in uploaded_files:
        file_name = getattr(file_obj, "name", "")
        file_bytes = file_obj.getvalue()
        hash_obj.update(str(file_name).encode("utf-8", errors="ignore"))
        hash_obj.update(file_bytes)
    return hash_obj.hexdigest()


def main() -> None:
    st.set_page_config(page_title="Finger Checker", layout="wide")
    AUTH.require_login()
    st.title("Finger Checker")
    st.caption("画像を1枚アップロードして、解析モード（手ごと/一括）を選んで人数・手・指を判定します。")
    api_key = get_default_api_key()

    with st.sidebar:
        if st.button("ログアウト", use_container_width=True):
            AUTH.logout()
            st.stop()
        st.subheader("Gemini設定")
        analysis_model = st.text_input("解析モデル", value=DEFAULT_ANALYSIS_MODEL)
        analysis_mode_label = st.radio(
            "解析モード",
            options=["手ごと厳密（高精度・遅い）", "一括解析（高速）"],
            index=0,
            horizontal=False,
        )
        analysis_mode = "per_hand" if analysis_mode_label.startswith("手ごと") else "batch"
        enable_marking = st.toggle("異常箇所を赤丸で表示", value=True)
        opencv_available = cv2 is not None and np is not None
        enable_opencv_bbox = st.toggle(
            "OpenCVで修正用bboxを補助検出",
            value=opencv_available,
            disabled=not opencv_available,
        )
        if not opencv_available:
            st.caption("OpenCV未インストールのため、bbox補助検出は無効です。")
        enable_fix = st.toggle("Nanobananaで指を修正", value=True)
        edit_model = st.text_input(
            "画像編集モデル",
            value=DEFAULT_EDIT_MODEL,
            disabled=not enable_fix,
        )
        output_size_label = st.radio(
            "Nanobanana出力サイズ",
            options=list(OUTPUT_LONG_EDGE_BY_LABEL.keys()),
            index=1,
            disabled=not enable_fix,
        )
        output_long_edge = OUTPUT_LONG_EDGE_BY_LABEL.get(output_size_label, 4096)
        extra_instruction = st.text_area(
            "Nanobanana追加指示（任意）",
            value=st.session_state.get("nanobanana_extra_instruction", ""),
            key="nanobanana_extra_instruction",
            disabled=not enable_fix,
            placeholder="例: 指輪などの装飾は変えずに、指の本数だけ修正してください。",
        )

    uploaded_files = st.file_uploader(
        "画像をアップロード（複数可）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("画像をアップロードすると解析を開始できます。")
        return

    if len(uploaded_files) > 1:
        st.info(f"{len(uploaded_files)}枚の画像を検出しました。順次処理できます。")
        batch_digest = build_uploaded_files_digest(uploaded_files)
        if st.session_state.get("batch_digest") != batch_digest:
            st.session_state["batch_digest"] = batch_digest
            st.session_state["batch_results"] = None

        if st.button("複数画像を順次処理", type="primary", use_container_width=True):
            if not api_key.strip():
                st.error("GEMINI_API_KEY を .streamlit/secrets.toml または環境変数で設定してください。")
            else:
                total = len(uploaded_files)
                progress = st.progress(0.0, text="処理を開始します...")
                batch_results: list[dict[str, Any]] = []
                for index, file_obj in enumerate(uploaded_files, start=1):
                    progress.progress((index - 1) / total, text=f"{index}/{total} 枚目を処理中: {file_obj.name}")
                    image_bytes = file_obj.getvalue()
                    mime_type = file_obj.type or "image/png"
                    try:
                        pipeline = run_pipeline_for_image(
                            api_key=api_key,
                            analysis_model=analysis_model,
                            analysis_mode=analysis_mode,
                            image_bytes=image_bytes,
                            mime_type=mime_type,
                            enable_marking=enable_marking,
                            enable_opencv_bbox=enable_opencv_bbox,
                            enable_fix=enable_fix,
                            edit_model=edit_model,
                            output_long_edge=output_long_edge,
                            extra_instruction=extra_instruction,
                        )
                        batch_results.append(
                            {
                                "file_name": file_obj.name,
                                "image_bytes": image_bytes,
                                "mime_type": mime_type,
                                "ok": True,
                                **pipeline,
                            }
                        )
                    except Exception as error:
                        batch_results.append(
                            {
                                "file_name": file_obj.name,
                                "image_bytes": image_bytes,
                                "mime_type": mime_type,
                                "ok": False,
                                "error": str(error),
                            }
                        )
                progress.progress(1.0, text="全画像の処理が完了しました。")
                st.session_state["batch_results"] = batch_results

        batch_results = st.session_state.get("batch_results")
        if isinstance(batch_results, list):
            st.subheader("複数画像の処理結果")
            for index, result in enumerate(batch_results, start=1):
                file_name = str(result.get("file_name", f"image_{index}"))
                with st.expander(f"{index}. {file_name}", expanded=False):
                    if not result.get("ok"):
                        st.error(f"処理失敗: {result.get('error', 'unknown error')}")
                        continue

                    analysis_result_multi = result.get("analysis_result", {})
                    mode_text_multi = (
                        "手ごと厳密（高精度・遅い）"
                        if analysis_result_multi.get("analysis_mode") == "per_hand"
                        else "一括解析（高速）"
                    )
                    st.caption(f"解析モード: {mode_text_multi}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("人数", analysis_result_multi.get("people_count", 0))
                    col2.metric("手の数", analysis_result_multi.get("hands_count", 0))
                    col3.metric("指の数", analysis_result_multi.get("fingers_count", 0))
                    st.write(f"要約: {analysis_result_multi.get('summary', '')}")

                    hands_multi = analysis_result_multi.get("hands", [])
                    if isinstance(hands_multi, list) and hands_multi:
                        st.write("手ごとの内訳")
                        st.dataframe(hands_multi, use_container_width=True)
                        anomalies_multi = detect_anomalies(
                            [h for h in hands_multi if isinstance(h, dict)]
                        )
                        if anomalies_multi:
                            st.warning(
                                f"5本未確認（または不確実）な手があります: {len(anomalies_multi)}件"
                            )
                        else:
                            st.success("全ての手で5本が明確に確認できました。")

                    anomaly_marks_multi = result.get("anomaly_marks")
                    if isinstance(anomaly_marks_multi, list) and anomaly_marks_multi:
                        st.info(f"赤丸候補: {len(anomaly_marks_multi)}箇所")
                        with st.expander("異常箇所の詳細（赤丸座標）", expanded=False):
                            st.dataframe(anomaly_marks_multi, use_container_width=True)
                    marked_image_multi = result.get("marked_image")
                    if marked_image_multi:
                        st.image(
                            marked_image_multi,
                            caption="赤丸は指の異常候補位置",
                            use_container_width=True,
                        )

                    fix_boxes_multi = result.get("fix_boxes")
                    if isinstance(fix_boxes_multi, list) and fix_boxes_multi:
                        method_multi = (
                            "OpenCV補助あり" if result.get("opencv_used") else "モデル推定bbox"
                        )
                        st.info(f"修正用bbox: {len(fix_boxes_multi)}件（{method_multi}）")
                        with st.expander("修正用bboxの詳細", expanded=False):
                            st.dataframe(fix_boxes_multi, use_container_width=True)
                    fix_boxes_preview_multi = result.get("fix_boxes_preview")
                    if fix_boxes_preview_multi:
                        st.image(
                            fix_boxes_preview_multi,
                            caption="オレンジ枠は指修正の対象範囲",
                            use_container_width=True,
                        )

                    if result.get("auto_fix_skipped_reason"):
                        st.warning(f"Nanobanana自動生成をスキップ: {result['auto_fix_skipped_reason']}")

                    edited_image_multi = result.get("edited_image")
                    if edited_image_multi:
                        st.image(edited_image_multi, caption="補正後画像", use_container_width=True)
                        st.download_button(
                            f"{file_name} をダウンロード",
                            data=edited_image_multi,
                            file_name=f"fixed_{file_name}",
                            mime=result.get("edited_mime", "image/png"),
                            key=f"download_fixed_{index}_{file_name}",
                        )
                    if enable_fix:
                        if st.button(
                            "Nanobananaで再生成",
                            key=f"regen_batch_{index}_{file_name}",
                            use_container_width=True,
                        ):
                            if not api_key.strip():
                                st.error(
                                    "補正前に GEMINI_API_KEY を .streamlit/secrets.toml または環境変数で設定してください。"
                                )
                            else:
                                source_bytes = result.get("image_bytes")
                                source_mime = str(result.get("mime_type") or "image/png")
                                if not isinstance(source_bytes, (bytes, bytearray)):
                                    st.error("元画像データが見つからないため再生成できませんでした。")
                                else:
                                    with st.spinner(f"{file_name} を再生成中..."):
                                        try:
                                            analysis_result_for_edit = result.get("analysis_result")
                                            fix_boxes_for_edit = result.get("fix_boxes")

                                            if not isinstance(analysis_result_for_edit, dict):
                                                analysis_result_for_edit = None
                                            if not isinstance(fix_boxes_for_edit, list):
                                                fix_boxes_for_edit = []

                                            if analysis_result_for_edit is None or not fix_boxes_for_edit:
                                                pipeline_refresh = run_pipeline_for_image(
                                                    api_key=api_key,
                                                    analysis_model=analysis_model,
                                                    analysis_mode=analysis_mode,
                                                    image_bytes=bytes(source_bytes),
                                                    mime_type=source_mime,
                                                    enable_marking=enable_marking,
                                                    enable_opencv_bbox=enable_opencv_bbox,
                                                    enable_fix=False,
                                                    edit_model=edit_model,
                                                    output_long_edge=output_long_edge,
                                                    extra_instruction=extra_instruction,
                                                )
                                                result["analysis_result"] = pipeline_refresh["analysis_result"]
                                                result["analysis_raw"] = pipeline_refresh["analysis_raw"]
                                                result["anomaly_marks"] = pipeline_refresh["anomaly_marks"]
                                                result["fix_boxes"] = pipeline_refresh["fix_boxes"]
                                                result["opencv_used"] = pipeline_refresh["opencv_used"]
                                                result["marked_image"] = pipeline_refresh["marked_image"]
                                                result["fix_boxes_preview"] = pipeline_refresh["fix_boxes_preview"]
                                                analysis_result_for_edit = pipeline_refresh["analysis_result"]
                                                fix_boxes_for_edit = pipeline_refresh["fix_boxes"]

                                            if not isinstance(fix_boxes_for_edit, list) or not fix_boxes_for_edit:
                                                st.warning("修正対象bboxが見つからないため、再生成をスキップしました。")
                                            else:
                                                edited_image, edited_mime, edit_raw = edit_image_with_nanobanana(
                                                    api_key=api_key,
                                                    model_name=edit_model,
                                                    image_bytes=bytes(source_bytes),
                                                    mime_type=source_mime,
                                                    analysis_json=analysis_result_for_edit,
                                                    fix_boxes=fix_boxes_for_edit,
                                                    output_long_edge=output_long_edge,
                                                    extra_instruction=extra_instruction,
                                                )
                                                result["edited_image"] = edited_image
                                                result["edited_mime"] = edited_mime or "image/png"
                                                result["edit_raw"] = edit_raw
                                                result["auto_fix_skipped_reason"] = None
                                                result["ok"] = True
                                                st.session_state["batch_results"] = batch_results
                                                st.rerun()
                                        except Exception as error:
                                            st.error(f"再生成に失敗しました: {error}")
        return

    uploaded_file = uploaded_files[0]
    image_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or "image/png"
    ensure_file_state(image_bytes)

    st.image(image_bytes, caption=f"入力画像: {uploaded_file.name}", use_container_width=True)

    if st.button("人数・手・指を解析", type="primary", use_container_width=True):
        if not api_key.strip():
            st.error(
                "GEMINI_API_KEY を .streamlit/secrets.toml または環境変数で設定してください。"
            )
        else:
            spinner_text = (
                "Geminiで厳密解析中（手ごと再判定）..."
                if analysis_mode == "per_hand"
                else "Geminiで一括解析中..."
            )
            if enable_fix:
                spinner_text = f"{spinner_text} / Nanobanana生成（{output_size_label}）..."
            with st.spinner(spinner_text):
                try:
                    pipeline = run_pipeline_for_image(
                        api_key=api_key,
                        analysis_model=analysis_model,
                        analysis_mode=analysis_mode,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        enable_marking=enable_marking,
                        enable_opencv_bbox=enable_opencv_bbox,
                        enable_fix=enable_fix,
                        edit_model=edit_model,
                        output_long_edge=output_long_edge,
                        extra_instruction=extra_instruction,
                    )
                    st.session_state["analysis_result"] = pipeline["analysis_result"]
                    st.session_state["analysis_raw"] = pipeline["analysis_raw"]
                    st.session_state["anomaly_marks"] = pipeline["anomaly_marks"]
                    st.session_state["fix_boxes"] = pipeline["fix_boxes"]
                    st.session_state["opencv_used"] = pipeline["opencv_used"]
                    st.session_state["marked_image"] = pipeline["marked_image"]
                    st.session_state["fix_boxes_preview"] = pipeline["fix_boxes_preview"]
                    st.session_state["edited_image"] = pipeline["edited_image"]
                    st.session_state["edited_mime"] = pipeline["edited_mime"] or "image/png"
                    st.session_state["edit_raw"] = pipeline["edit_raw"]
                    if pipeline["auto_fix_skipped_reason"]:
                        st.warning(
                            f"Nanobanana自動生成をスキップしました（{pipeline['auto_fix_skipped_reason']}）"
                        )
                except Exception as error:
                    st.error(f"解析に失敗しました: {error}")

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        st.subheader("解析結果")
        mode_text = (
            "手ごと厳密（高精度・遅い）"
            if analysis_result.get("analysis_mode") == "per_hand"
            else "一括解析（高速）"
        )
        st.caption(f"解析モード: {mode_text}")
        col1, col2, col3 = st.columns(3)
        col1.metric("人数", analysis_result.get("people_count", 0))
        col2.metric("手の数", analysis_result.get("hands_count", 0))
        col3.metric("指の数", analysis_result.get("fingers_count", 0))

        hands = analysis_result.get("hands", [])
        if hands:
            st.write("手ごとの内訳")
            st.dataframe(hands, use_container_width=True)

        anomalies = detect_anomalies([h for h in hands if isinstance(h, dict)])
        anomaly_marks = st.session_state.get("anomaly_marks")
        if not isinstance(anomaly_marks, list):
            anomaly_marks = collect_anomaly_marks(analysis_result)
            st.session_state["anomaly_marks"] = anomaly_marks
        fix_boxes = st.session_state.get("fix_boxes")
        if not isinstance(fix_boxes, list):
            fix_boxes, opencv_used = collect_fix_boxes(
                analysis_result,
                image_bytes=image_bytes,
                use_opencv_refine=enable_opencv_bbox,
            )
            st.session_state["fix_boxes"] = fix_boxes
            st.session_state["opencv_used"] = opencv_used

        if anomalies:
            st.warning(f"5本未確認（または不確実）な手があります: {len(anomalies)}件")
        else:
            st.success("全ての手で5本が明確に確認できました。")

        if anomaly_marks:
            st.info(f"丸付け候補: {len(anomaly_marks)}箇所")
            with st.expander("異常箇所の詳細（丸付け座標）"):
                st.dataframe(anomaly_marks, use_container_width=True)

        opencv_used = bool(st.session_state.get("opencv_used"))
        if fix_boxes:
            method = "OpenCV補助あり" if opencv_used else "モデル推定bbox"
            st.info(f"修正用bbox: {len(fix_boxes)}件（{method}）")
            with st.expander("修正用bboxの詳細"):
                st.dataframe(fix_boxes, use_container_width=True)

            fix_boxes_preview = st.session_state.get("fix_boxes_preview")
            if not fix_boxes_preview:
                try:
                    fix_boxes_preview = build_box_preview_image(image_bytes, fix_boxes)
                    st.session_state["fix_boxes_preview"] = fix_boxes_preview
                except Exception as error:
                    st.error(f"bboxプレビュー画像の生成に失敗しました: {error}")
                    fix_boxes_preview = None

            if fix_boxes_preview:
                st.subheader("修正対象bboxプレビュー")
                st.image(fix_boxes_preview, caption="オレンジ枠は指修正の対象範囲", use_container_width=True)

        if analysis_result.get("summary"):
            st.write(f"要約: {analysis_result['summary']}")
        if analysis_result.get("detection_summary"):
            st.caption(f"手検出メモ: {analysis_result['detection_summary']}")

        if enable_marking and anomaly_marks:
            marked_image = st.session_state.get("marked_image")
            if not marked_image:
                try:
                    marked_image = build_marked_image(image_bytes, anomaly_marks)
                    st.session_state["marked_image"] = marked_image
                except Exception as error:
                    st.error(f"丸付け画像の生成に失敗しました: {error}")
                    marked_image = None

            if marked_image:
                st.subheader("異常箇所の丸付け画像")
                st.image(marked_image, caption="赤丸は指の異常候補位置", use_container_width=True)

    if enable_fix:
        st.subheader("Nanobanana 指補正")
        analysis_result_for_edit = analysis_result if isinstance(analysis_result, dict) else None
        fix_boxes_for_edit = st.session_state.get("fix_boxes")
        if not isinstance(fix_boxes_for_edit, list):
            fix_boxes_for_edit = []
        if analysis_result_for_edit is None:
            st.info("解析結果がありません。再生成ボタンを押すと解析と補正を続けて実行します。")
        elif not fix_boxes_for_edit:
            st.warning("修正対象bboxが見つかっていません。解析を再実行するか、画像を見直してください。")
        st.caption(f"出力サイズ: {output_size_label}")
        if st.button("Nanobananaで再生成", use_container_width=True):
            if not api_key.strip():
                st.error("補正前に GEMINI_API_KEY を .streamlit/secrets.toml または環境変数で設定してください。")
            else:
                with st.spinner("Nanobananaで再生成中..."):
                    try:
                        if analysis_result_for_edit is None:
                            pipeline = run_pipeline_for_image(
                                api_key=api_key,
                                analysis_model=analysis_model,
                                analysis_mode=analysis_mode,
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                                enable_marking=enable_marking,
                                enable_opencv_bbox=enable_opencv_bbox,
                                enable_fix=False,
                                edit_model=edit_model,
                                output_long_edge=output_long_edge,
                                extra_instruction=extra_instruction,
                            )
                            st.session_state["analysis_result"] = pipeline["analysis_result"]
                            st.session_state["analysis_raw"] = pipeline["analysis_raw"]
                            st.session_state["anomaly_marks"] = pipeline["anomaly_marks"]
                            st.session_state["fix_boxes"] = pipeline["fix_boxes"]
                            st.session_state["opencv_used"] = pipeline["opencv_used"]
                            st.session_state["marked_image"] = pipeline["marked_image"]
                            st.session_state["fix_boxes_preview"] = pipeline["fix_boxes_preview"]
                            analysis_result_for_edit = pipeline["analysis_result"]
                            fix_boxes_for_edit = pipeline["fix_boxes"] if isinstance(
                                pipeline.get("fix_boxes"), list
                            ) else []

                        if not fix_boxes_for_edit:
                            st.warning("修正対象bboxが見つからないため、再生成をスキップしました。")
                        else:
                            edited_image, edited_mime, edit_raw = edit_image_with_nanobanana(
                                api_key=api_key,
                                model_name=edit_model,
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                                analysis_json=analysis_result_for_edit,
                                fix_boxes=fix_boxes_for_edit,
                                output_long_edge=output_long_edge,
                                extra_instruction=extra_instruction,
                            )
                            st.session_state["edited_image"] = edited_image
                            st.session_state["edited_mime"] = edited_mime or "image/png"
                            st.session_state["edit_raw"] = edit_raw
                    except Exception as error:
                        st.error(f"補正に失敗しました: {error}")

        if st.session_state.get("edited_image"):
            st.image(
                st.session_state["edited_image"],
                caption="補正後画像",
                use_container_width=True,
            )
            st.download_button(
                "補正後画像をダウンロード",
                data=st.session_state["edited_image"],
                file_name=f"fixed_{uploaded_file.name}",
                mime=st.session_state.get("edited_mime", "image/png"),
            )


if __name__ == "__main__":
    main()
