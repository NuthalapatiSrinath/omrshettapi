#!/usr/bin/env python3
"""
omr_reader.py
Robust OMR reader that:
 - Detects and deskews page
 - Extracts MCQ answers from a multi-column grid
 - Extracts integer-type answers (box digits or OCR)
 - Produces debug images in ./debug/
 - CLI options to configure grid and integer fields

Usage:
  py python_scripts\omr_reader.py "C:\path\to\img.jpg" --rows 20 --cols 5 --options 4 --debug

Example intspec:
  --intspec "91:5,92:3,93:5"   (question#:digitCount)

Notes:
 - Requires tesseract installed and on PATH. If not, set pytesseract.pytesseract.tesseract_cmd
 - Tune thresholds to match your template
"""

import sys
import os
import json
import argparse
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract

# If Tesseract is not in PATH on Windows, uncomment & set:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    """Robust read for Unicode/Windows paths using np.fromfile + imdecode"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def resize_for_processing(img: np.ndarray, max_width=1400) -> np.ndarray:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_page_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest 4-point contour approximating the sheet border"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx.reshape(4, 2)
    return None


def crop_roi(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(0, min(w, w_img - x))
    h = max(0, min(h, h_img - y))
    return img[y:y + h, x:x + w]


def make_circular_mask(h: int, w: int, ratio=0.36) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * ratio)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def preprocess_zone_for_detection(zone_bgr: np.ndarray) -> np.ndarray:
    """Return binary inverted zone where filled areas are white (255)"""
    zone_gray = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
    # equalize to reduce lighting differences
    zone_gray = cv2.equalizeHist(zone_gray)
    _, th = cv2.threshold(zone_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # small open to remove dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th


def detect_mcq_grid(grid_img: np.ndarray,
                    rows: int,
                    cols: int,
                    options_per_q: int,
                    circle_ratio=0.36,
                    min_fill=0.30,
                    gap=0.18,
                    debug_prefix="grid") -> List[Dict]:
    """
    grid_img: BGR image cropping the bubble area only
    returns list of { q: int, selected: 'A'|'B'|..., scores: [..], raw_scores: [..] }
    """

    H, W = grid_img.shape[:2]
    cell_h = H // rows
    cell_w = W // cols
    results = []
    qnum = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Save a threshold debug image for grid
    grid_gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    grid_th = cv2.adaptiveThreshold(grid_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 2)
    debug_th_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_adaptive_th.png")
    cv2.imwrite(debug_th_path, grid_th)

    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_h
            cell = grid_img[y:y + cell_h, x:x + cell_w]
            if cell.size == 0:
                results.append({'q': qnum, 'selected': None, 'scores': [], 'raw': []})
                qnum += 1
                continue

            # We'll split each cell horizontally into options_per_q zones
            zone_w = cell.shape[1] // options_per_q
            raw_scores = []
            # process each zone
            for opt in range(options_per_q):
                zx = opt * zone_w
                zone = cell[:, zx: zx + zone_w]
                if zone.size == 0:
                    raw_scores.append(0.0)
                    continue
                # preprocess and mask
                zone_th = preprocess_zone_for_detection(zone)
                # apply circular mask relative to zone size
                mask = make_circular_mask(zone_th.shape[0], zone_th.shape[1], ratio=circle_ratio)
                masked = cv2.bitwise_and(zone_th, zone_th, mask=mask)
                filled = cv2.countNonZero(masked)
                area = cv2.countNonZero(mask)
                ratio_filled = (filled / area) if area > 0 else 0.0
                raw_scores.append(ratio_filled)

            # Normalize raw_scores
            arr = np.array(raw_scores, dtype=float)
            if arr.size == 0:
                results.append({'q': qnum, 'selected': None, 'scores': [], 'raw': raw_scores})
                qnum += 1
                continue
            # scale between 0-1 (relative)
            minv = arr.min()
            maxv = arr.max()
            norm = (arr - minv) / (maxv - minv + 1e-9)

            # Determine selection with stricter checks
            best_idx = int(np.argmax(norm))
            best_val = float(norm[best_idx])
            second_val = float(np.partition(norm, -2)[-2]) if norm.size >= 2 else 0.0

            selected_letter = None
            # rules:
            # - absolute raw fill (not normalized) must exceed min_fill (e.g., 0.30)
            # - normalized best gap must be greater than gap threshold
            # - if ambiguous (close), mark None
            if raw_scores[best_idx] >= min_fill and (best_val - second_val) >= gap:
                selected_letter = chr(ord('A') + best_idx)
            else:
                selected_letter = None

            results.append({'q': qnum, 'selected': selected_letter, 'scores': norm.tolist(), 'raw': raw_scores})
            qnum += 1

    # save overlay for debug
    overlay = grid_img.copy()
    H, W = overlay.shape[:2]
    for item in results:
        idx = item['q'] - 1
        rr = idx // cols
        cc = idx % cols
        x = cc * (W // cols)
        y = rr * (H // rows)
        w = (W // cols) - 1
        h = (H // rows) - 1
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), 1)
        if item['selected'] is not None:
            sel = ord(item['selected']) - ord('A')
            zone_w = w // options_per_q
            cx = int(x + sel * zone_w + zone_w / 2)
            cy = int(y + h / 2)
            cv2.circle(overlay, (cx, cy), int(min(zone_w, h) * 0.18), (0, 200, 0), 2)
        cv2.putText(overlay, str(item['q']), (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

    debug_overlay_path = os.path.join(DEBUG_DIR, f"{debug_prefix}_overlay.png")
    cv2.imwrite(debug_overlay_path, overlay)

    return results


def read_integer_box_ocr(region_bgr: np.ndarray, expected_digits: int) -> str:
    """
    Uses pytesseract to read digits. Preprocess: threshold, resize.
    Returns raw string (digits only) trimmed/padded to expected_digits if possible.
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    # resize for better OCR
    H, W = gray.shape[:2]
    scale = max(1.0, 600.0 / max(W, H))
    gray = cv2.resize(gray, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
    # threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert so text is black on white (pytesseract prefers)
    th_inv = 255 - th
    pil = Image.fromarray(th_inv)
    # psm 7 (single text line) or 8 (single word) and digits only
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(pil, config=config)
    # clean
    digits = "".join([c for c in text if c.isdigit()])
    # if too long, take last expected_digits (common for roll)
    if len(digits) > expected_digits:
        digits = digits[-expected_digits:]
    return digits


def read_integer_box_bubble_style(region_bgr: np.ndarray, digits: int, per_digit_cols=10) -> Optional[str]:
    """
    If integer region is bubble-based (each digit is a small grid), we can attempt to detect dark zones.
    This is a heuristic: it splits the region horizontally into `digits` columns and per column into `per_digit_cols`
    possible filled zones (0-9).
    """
    H, W = region_bgr.shape[:2]
    if H == 0 or W == 0:
        return None
    col_w = W // digits
    result_digits = []
    for d in range(digits):
        x = d * col_w
        digit_zone = region_bgr[:, x: x + col_w]
        zh, zw = digit_zone.shape[:2]
        if zh == 0 or zw == 0:
            result_digits.append(None)
            continue
        # split vertically into per_digit_cols zones (0..9)
        zone_h = zh // per_digit_cols
        scores = []
        for i in range(per_digit_cols):
            zy = i * zone_h
            zone = digit_zone[zy: zy + zone_h, :]
            zone_th = preprocess_zone_for_detection(zone)
            filled = cv2.countNonZero(zone_th)
            area = zone_th.shape[0] * zone_th.shape[1]
            scores.append((filled / max(area, 1)))
        arr = np.array(scores)
        # find best
        best = int(np.argmax(arr))
        # require a minimum fill and a gap to second best
        best_val = float(arr[best])
        second = float(np.partition(arr, -2)[-2]) if arr.size >= 2 else 0.0
        if best_val > 0.25 and (best_val - second) > 0.12:
            result_digits.append(str(best))
        else:
            result_digits.append(None)
    # build string if all digits found
    if any(d is None for d in result_digits):
        # partial fallback - join known digits with blanks
        out = "".join([d if d is not None else "" for d in result_digits])
    else:
        out = "".join(result_digits)
    return out


def parse_intspec(intspec: str) -> Dict[int, int]:
    """
    intspec format: "91:5,92:3,93:5"
    returns {91:5, 92:3, 93:5}
    """
    mapping = {}
    if not intspec:
        return mapping
    parts = [p.strip() for p in intspec.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            q, d = p.split(":", 1)
            try:
                mapping[int(q)] = int(d)
            except Exception:
                continue
    return mapping


def main_cli():
    parser = argparse.ArgumentParser(description="Robust OMR Reader")
    parser.add_argument("image", help="path to image")
    parser.add_argument("--rows", type=int, default=20, help="rows per column in MCQ grid")
    parser.add_argument("--cols", type=int, default=5, help="number of columns of MCQ grid")
    parser.add_argument("--options", type=int, default=4, help="options per question (e.g. 4 for A-D)")
    parser.add_argument("--grid", nargs=4, type=float, metavar=("x", "y", "w", "h"),
                        help="grid bbox as fractions of warped image: x y w h (0..1)")
    parser.add_argument("--min_fill", type=float, default=0.30, help="minimum raw fill ratio to accept a bubble")
    parser.add_argument("--gap", type=float, default=0.18, help="minimum normalized gap between top and second")
    parser.add_argument("--intspec", type=str, default="", help="integer fields spec: '91:5,92:3' question#:digits")
    parser.add_argument("--intboxes", type=str, default="", help="optional integer boxes bboxes as csv: q:x:y:w:h;... (pixel units after warp)")
    parser.add_argument("--debug", action="store_true", help="save debug images to debug/")
    parser.add_argument("--deskewless", action="store_true", help="skip contour-based deskewing (use original image)")
    parser.add_argument("--resize", type=int, default=1400, help="max width to resize for processing")
    args = parser.parse_args()

    image_path = args.image
    rows = args.rows
    cols = args.cols
    options_per_q = args.options
    grid_frac = args.grid
    min_fill = args.min_fill
    gap = args.gap
    intspec = parse_intspec(args.intspec)
    intboxes = {}  # parse intboxes if provided: "q:x:y:w:h;q2:..."
    if args.intboxes:
        for seg in args.intboxes.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            try:
                qstr, x, y, w, h = seg.split(":")
                intboxes[int(qstr)] = (int(x), int(y), int(w), int(h))
            except Exception:
                continue

    # Read image
    try:
        raw = read_image(image_path)
    except Exception as e:
        print(json.dumps({"success": False, "message": str(e)}))
        sys.exit(1)

    # Resize for stability
    proc = resize_for_processing(raw, max_width=args.resize)

    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    page_cnt = None
    if not args.deskewless:
        page_cnt = find_page_contour(gray)

    if page_cnt is not None:
        warped = four_point_transform(proc, page_cnt)
    else:
        warped = proc.copy()

    if args.debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "warped_full.png"), warped)

    H, W = warped.shape[:2]

    # Compute grid bbox in pixels
    if grid_frac and len(grid_frac) == 4:
        gx = int(grid_frac[0] * W)
        gy = int(grid_frac[1] * H)
        gw = int(grid_frac[2] * W)
        gh = int(grid_frac[3] * H)
    else:
        # default heuristic for many OMR templates: central grid region
        gx = int(W * 0.055)
        gy = int(H * 0.16)
        gw = int(W * 0.9)
        gh = int(H * 0.72)

    grid_img = crop_roi(warped, (gx, gy, gw, gh))
    if args.debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "grid_crop.png"), grid_img)

    # Detect MCQ answers
    mcq_results = detect_mcq_grid(grid_img, rows, cols, options_per_q,
                                  circle_ratio=0.36,
                                  min_fill=min_fill,
                                  gap=gap,
                                  debug_prefix="grid")

    # Convert q numbers to absolute numbering across columns:
    # Our detector enumerates left-to-right across columns, top-to-bottom per column.
    # If your sheet numbers differently, you may need to reorder accordingly.
    # The detect_mcq_grid uses q starting at 1 already sequentially.

    # Read roll number via OCR from a top area (adjust heuristics)
    # We'll look at the top center-left area as typical roll is present there
    roll_bbox = (int(W * 0.08), int(H * 0.02), int(W * 0.28), int(H * 0.08))
    roll_region = crop_roi(warped, roll_bbox)
    if args.debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, "roll_region.png"), roll_region)

    # OCR roll as digits only
    try:
        roll_text = read_integer_box_ocr(roll_region, expected_digits=10)
    except Exception:
        roll_text = ""

    # Integer fields: two ways:
    # 1) intspec provides question# : digits (we try to find their position by heuristics)
    # 2) if intboxes provided, use exact pixel boxes after warp
    integer_results = []

    # Heuristic int box layout: if your sheet has bottom boxes (like 91..95), you can calculate locations
    # We'll attempt to locate integer boxes by scanning for small rectangular boxes in the bottom area if intboxes absent.
    if intboxes:
        # use explicit boxes
        for qnum, bbox in intboxes.items():
            x, y, w, h = bbox
            region = crop_roi(warped, (x, y, w, h))
            digits = intspec.get(qnum, None) or 0
            val = ""

            # first try OCR
            if digits and digits > 0:
                val = read_integer_box_ocr(region, digits)
                if not val or len(val) < digits:
                    # fallback to bubble style detection
                    val = read_integer_box_bubble_style(region, digits, per_digit_cols=10) or ""
            integer_results.append({"q": qnum, "value": val})
    else:
        # try intspec mapping and heuristic bottom area scanning
        if intspec:
            # We'll assume integer boxes lie in a bottom region (last 0.15 of sheet)
            bottom_region = crop_roi(warped, (int(W * 0.05), int(H * 0.78), int(W * 0.9), int(H * 0.2)))
            if args.debug:
                cv2.imwrite(os.path.join(DEBUG_DIR, "int_bottom_region.png"), bottom_region)
            # create a naive grid split based on how many integer fields we have and their order
            # Sort intspec by key ascending
            items = sorted(intspec.items(), key=lambda x: x[0])
            n = len(items)
            # arrange horizontally if many else vertically attempt
            # We'll split bottom region into n equal horizontal slices (this may not match all templates!)
            hbr, wbr = bottom_region.shape[:2]
            for idx, (qnum, digits) in enumerate(items):
                # slice horizontally left->right
                single_w = wbr // n
                sx = idx * single_w
                region = bottom_region[:, sx: sx + single_w]
                # try OCR, fallback to bubble detection
                val = read_integer_box_ocr(region, digits)
                if not val or len(val) < digits:
                    fb = read_integer_box_bubble_style(region, digits, per_digit_cols=10)
                    if fb:
                        val = fb
                integer_results.append({"q": qnum, "value": val})
        # else skip

    # Build output
    out_questions = []
    for item in mcq_results:
        out_questions.append({"q": int(item["q"]), "selected": item["selected"], "scores": item["scores"], "raw": item["raw"]})

    output = {
        "success": True,
        "roll": roll_text if roll_text else None,
        "questions": out_questions,
        "integers": integer_results,
        "meta": {
            "grid_bbox": [gx, gy, gw, gh],
            "warp_shape": [W, H],
            "rows": rows,
            "cols": cols,
            "options_per_q": options_per_q
        }
    }

    # Save a final overlay image with detected selections for debugging
    try:
        overlay_all = warped.copy()
        # grid overlay
        gh_img = grid_img.copy()
        # draw rect of grid on overlay_all
        cv2.rectangle(overlay_all, (gx, gy), (gx + gw, gy + gh), (0, 120, 255), 2)
        # write summary
        cv2.putText(overlay_all, f"MCQ detected: {len(out_questions)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 180, 20), 2)
        cv2.imwrite(os.path.join(DEBUG_DIR, "overlay_all.png"), overlay_all)
    except Exception:
        pass

    print(json.dumps(output, ensure_ascii=False, indent=None))


if __name__ == "__main__":
    main_cli()
 