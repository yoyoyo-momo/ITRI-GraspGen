import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class DetectionResult:
    image_name: str
    handle_side: str
    center_x: int | None
    center_y: int | None
    confidence: float
    note: str


@dataclass
class DetectionOverlay:
    teapot_bbox: tuple[int, int, int, int] | None
    hole_bbox: tuple[int, int, int, int] | None
    hole_center: tuple[int, int] | None


class TeapotHandleDetector:
    def __init__(
        self,
        target_max_dim: int = 1400,
        auto_refine: bool = False,
        refine_confidence: float = 0.60,
    ) -> None:
        self.target_max_dim = target_max_dim
        self.auto_refine = auto_refine
        self.refine_confidence = refine_confidence

    @staticmethod
    def _binary_dilate(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        windows = [
            padded[i : i + mask.shape[0], j : j + mask.shape[1]]
            for i in range(3)
            for j in range(3)
        ]
        return np.logical_or.reduce(windows)

    @staticmethod
    def _binary_erode(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        windows = [
            padded[i : i + mask.shape[0], j : j + mask.shape[1]]
            for i in range(3)
            for j in range(3)
        ]
        return np.logical_and.reduce(windows)

    def _binary_open(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = mask.copy()
        for _ in range(iterations):
            out = self._binary_erode(out)
        for _ in range(iterations):
            out = self._binary_dilate(out)
        return out

    def _binary_close(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = mask.copy()
        for _ in range(iterations):
            out = self._binary_dilate(out)
        for _ in range(iterations):
            out = self._binary_erode(out)
        return out

    @staticmethod
    def _connected_components(mask: np.ndarray) -> list[dict]:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components: list[dict] = []

        ys, xs = np.where(mask)
        for y0, x0 in zip(ys, xs, strict=False):
            if visited[y0, x0]:
                continue

            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True

            coords_y: list[int] = []
            coords_x: list[int] = []

            while stack:
                y, x = stack.pop()
                coords_y.append(y)
                coords_x.append(x)

                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            y_arr = np.array(coords_y)
            x_arr = np.array(coords_x)
            components.append(
                {
                    "area": len(coords_y),
                    "ymin": int(y_arr.min()),
                    "ymax": int(y_arr.max()),
                    "xmin": int(x_arr.min()),
                    "xmax": int(x_arr.max()),
                    "cy": float(y_arr.mean()),
                    "cx": float(x_arr.mean()),
                    "pixels": list(zip(coords_y, coords_x, strict=False)),
                }
            )

        return components

    def _merge_nearby_components(
        self,
        mask: np.ndarray,
        max_gap_px: int = 10,
        min_area_px: int = 8,
        prefer_center: bool = True,
    ) -> np.ndarray:
        components = self._connected_components(mask)
        if not components:
            return np.zeros_like(mask, dtype=bool)

        if prefer_center:
            h, w = mask.shape
            cy0 = (h - 1) / 2.0
            cx0 = (w - 1) / 2.0

            def score(comp: dict) -> float:
                dy = (comp["cy"] - cy0) / max(1.0, h)
                dx = (comp["cx"] - cx0) / max(1.0, w)
                dist = float(np.hypot(dx, dy))
                center_bonus = max(0.0, 0.9 - dist)
                return float(comp["area"]) * (1.0 + center_bonus)

            main = max(components, key=score)
        else:
            main = max(components, key=lambda c: c["area"])

        out = np.zeros_like(mask, dtype=bool)
        for y, x in main["pixels"]:
            out[y, x] = True

        mymin, mymax = main["ymin"], main["ymax"]
        mxmin, mxmax = main["xmin"], main["xmax"]
        main_area = int(main["area"])
        dynamic_min_area = max(min_area_px, int(main_area * 0.0004))

        for comp in components:
            if comp is main or comp["area"] < dynamic_min_area:
                continue

            cymin, cymax = comp["ymin"], comp["ymax"]
            cxmin, cxmax = comp["xmin"], comp["xmax"]

            gap_x = max(0, max(mxmin - cxmax, cxmin - mxmax))
            gap_y = max(0, max(mymin - cymax, cymin - mymax))
            gap = float(np.hypot(gap_x, gap_y))

            if gap <= max_gap_px:
                for y, x in comp["pixels"]:
                    out[y, x] = True

        return out

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        total = gray.size
        if total == 0:
            return 127

        values = np.arange(256, dtype=np.float64)
        total_sum = float(np.dot(values, hist))

        sum_bg = 0.0
        weight_bg = 0.0
        max_var = -1.0
        threshold = 127

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (total_sum - sum_bg) / weight_fg
            between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

            if between_var > max_var:
                max_var = between_var
                threshold = t

        return int(threshold)

    @staticmethod
    def _extract_holes(foreground: np.ndarray) -> np.ndarray:
        background = ~foreground
        h, w = background.shape
        visited = np.zeros_like(background, dtype=bool)
        stack: list[tuple[int, int]] = []

        for x in range(w):
            if background[0, x]:
                stack.append((0, x))
            if background[h - 1, x]:
                stack.append((h - 1, x))
        for y in range(h):
            if background[y, 0]:
                stack.append((y, 0))
            if background[y, w - 1]:
                stack.append((y, w - 1))

        while stack:
            y, x = stack.pop()
            if visited[y, x] or not background[y, x]:
                continue
            visited[y, x] = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < h
                    and 0 <= nx < w
                    and not visited[ny, nx]
                    and background[ny, nx]
                ):
                    stack.append((ny, nx))

        holes = background & (~visited)
        return holes

    @staticmethod
    def _shift_components(
        components: list[dict], offset_y: int, offset_x: int
    ) -> list[dict]:
        shifted: list[dict] = []
        for comp in components:
            shifted.append(
                {
                    "area": comp["area"],
                    "ymin": comp["ymin"] + offset_y,
                    "ymax": comp["ymax"] + offset_y,
                    "xmin": comp["xmin"] + offset_x,
                    "xmax": comp["xmax"] + offset_x,
                    "cy": comp["cy"] + offset_y,
                    "cx": comp["cx"] + offset_x,
                }
            )
        return shifted

    def _make_foreground_mask(
        self, image_rgb: np.ndarray, target_max_dim: int = 1400
    ) -> np.ndarray:
        h, w, _ = image_rgb.shape
        max_dim = max(h, w)
        min_process_dim = 900

        if target_max_dim > 0 and max_dim > target_max_dim:
            scale = target_max_dim / float(max_dim)
        elif max_dim < min_process_dim:
            scale = min_process_dim / float(max_dim)
        else:
            scale = 1.0

        if scale != 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            small_rgb = np.array(
                Image.fromarray(image_rgb).resize(
                    (new_w, new_h), resample=Image.Resampling.BILINEAR
                )
            )
        else:
            small_rgb = image_rgb

        gray = (
            0.299 * small_rgb[:, :, 0].astype(np.float32)
            + 0.587 * small_rgb[:, :, 1].astype(np.float32)
            + 0.114 * small_rgb[:, :, 2].astype(np.float32)
        ).astype(np.uint8)

        threshold = self._otsu_threshold(gray)
        dark_bias = 10
        mask = gray < min(245, threshold + dark_bias)

        close_iters = 2 if scale <= 1.0 else 3
        mask = self._binary_close(mask, iterations=close_iters)
        mask = self._binary_open(mask, iterations=1)
        mask = self._merge_nearby_components(
            mask, max_gap_px=12 if scale > 1.0 else 10, min_area_px=6
        )
        mask = self._binary_close(mask, iterations=1)

        if scale != 1.0:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
            mask = np.array(mask_img) > 0
            mask = self._binary_close(mask, iterations=1)

        return mask

    @staticmethod
    def _to_rgb_uint8(image_array: np.ndarray) -> np.ndarray:
        arr = np.asarray(image_array)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        else:
            raise ValueError("image_array must be shape HxW, HxWx1, HxWx3, or HxWx4")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    @staticmethod
    def _should_refine(
        result: DetectionResult,
        auto_refine: bool,
        target_max_dim: int,
        threshold: float,
    ) -> bool:
        if not auto_refine or target_max_dim <= 0:
            return False
        return result.handle_side == "unknown" or result.confidence < threshold

    @staticmethod
    def _is_better_refinement(base: DetectionResult, refined: DetectionResult) -> bool:
        better_side = refined.handle_side != "unknown" and base.handle_side == "unknown"
        better_conf = refined.confidence > base.confidence + 0.05
        return better_side or better_conf

    @staticmethod
    def collect_images(path: Path) -> list[Path]:
        if path.is_file():
            return [path]

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])

    @staticmethod
    def render_annotated(
        image_rgb: np.ndarray, overlay: DetectionOverlay
    ) -> Image.Image:
        annotated = Image.fromarray(image_rgb, mode="RGB")
        draw = ImageDraw.Draw(annotated)

        if overlay.teapot_bbox is not None:
            xmin, ymin, xmax, ymax = overlay.teapot_bbox
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(0, 255, 0), width=3)

        if overlay.hole_bbox is not None:
            xmin, ymin, xmax, ymax = overlay.hole_bbox
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=3)

        if overlay.hole_center is not None:
            cx, cy = overlay.hole_center
            r = 5
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 0, 0))

        return annotated

    @staticmethod
    def save_outputs(
        output_dir: Path,
        image_path: Path,
        image_rgb: np.ndarray,
        overlay: DetectionOverlay,
        mask: np.ndarray,
    ) -> None:
        annotated = TeapotHandleDetector.render_annotated(image_rgb, overlay)
        out_path = output_dir / f"{image_path.stem}_handle{image_path.suffix}"
        annotated.save(out_path)

        mask_path = output_dir / f"{image_path.stem}_mask.png"
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image.save(mask_path)

    def detect(
        self,
        image_array: np.ndarray,
        image_name: str = "image",
        target_max_dim: int | None = None,
    ) -> tuple[DetectionResult, np.ndarray, DetectionOverlay]:
        run_target_max_dim = (
            self.target_max_dim if target_max_dim is None else target_max_dim
        )
        rgb = self._to_rgb_uint8(image_array)
        mask = self._make_foreground_mask(rgb, target_max_dim=run_target_max_dim)

        if mask.sum() < 300:
            result = DetectionResult(
                image_name=image_name,
                handle_side="unknown",
                center_x=None,
                center_y=None,
                confidence=0.0,
                note="Could not isolate teapot silhouette.",
            )
            overlay = DetectionOverlay(
                teapot_bbox=None, hole_bbox=None, hole_center=None
            )
            return result, mask, overlay

        components = self._connected_components(mask)
        teapot = max(components, key=lambda c: c["area"])
        ymin, ymax = teapot["ymin"], teapot["ymax"]
        xmin, xmax = teapot["xmin"], teapot["xmax"]
        teapot_crop = mask[ymin : ymax + 1, xmin : xmax + 1]

        holes_crop = self._extract_holes(teapot_crop)
        hole_components_crop = self._connected_components(holes_crop)
        hole_components = self._shift_components(
            hole_components_crop, offset_y=ymin, offset_x=xmin
        )

        min_hole_area = max(35, int(teapot["area"] * 0.0015))
        candidates = [c for c in hole_components if c["area"] >= min_hole_area]

        teapot_bbox = (teapot["xmin"], teapot["ymin"], teapot["xmax"], teapot["ymax"])

        if not candidates:
            result = DetectionResult(
                image_name=image_name,
                handle_side="unknown",
                center_x=None,
                center_y=None,
                confidence=0.25,
                note="No handle hole detected (likely front/back view or low contrast).",
            )
            overlay = DetectionOverlay(
                teapot_bbox=teapot_bbox, hole_bbox=None, hole_center=None
            )
            return result, mask, overlay

        handle_hole = max(candidates, key=lambda c: c["area"])
        cx = int(round(handle_hole["cx"]))
        cy = int(round(handle_hole["cy"]))
        handle_side = "left" if handle_hole["cx"] < teapot["cx"] else "right"

        hole_bbox = (
            handle_hole["xmin"],
            handle_hole["ymin"],
            handle_hole["xmax"],
            handle_hole["ymax"],
        )
        hole_center = (cx, cy)

        confidence = min(
            0.99, 0.55 + (handle_hole["area"] / max(1, teapot["area"])) * 8.0
        )
        result = DetectionResult(
            image_name=image_name,
            handle_side=handle_side,
            center_x=cx,
            center_y=cy,
            confidence=float(confidence),
            note="Handle hole detected.",
        )
        overlay = DetectionOverlay(
            teapot_bbox=teapot_bbox, hole_bbox=hole_bbox, hole_center=hole_center
        )
        return result, mask, overlay

    def detect_auto(
        self,
        image_array: np.ndarray,
        image_name: str = "image",
    ) -> tuple[DetectionResult, np.ndarray, DetectionOverlay]:
        result, mask, overlay = self.detect(image_array, image_name=image_name)

        if self._should_refine(
            result,
            auto_refine=self.auto_refine,
            target_max_dim=self.target_max_dim,
            threshold=self.refine_confidence,
        ):
            refined_result, refined_mask, refined_overlay = self.detect(
                image_array,
                image_name=image_name,
                target_max_dim=0,
            )
            if self._is_better_refinement(result, refined_result):
                result, mask, overlay = refined_result, refined_mask, refined_overlay
                result.note = f"{result.note} (refined)"

        return result, mask, overlay


def detect_handle(
    image_array: np.ndarray,
    image_name: str = "image",
    target_max_dim: int = 1400,
) -> tuple[DetectionResult, np.ndarray, DetectionOverlay]:
    detector = TeapotHandleDetector(target_max_dim=target_max_dim, auto_refine=False)
    return detector.detect(image_array=image_array, image_name=image_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect where the teapot handle is in image(s)."
    )
    parser.add_argument(
        "--input", type=str, default="sample", help="Input image file or folder."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output folder for annotated images and masks. If omitted, only CSV is printed.",
    )
    parser.add_argument(
        "--target-max-dim",
        type=int,
        default=1400,
        help="Max image dimension used for mask processing; higher means more detail but slower. Set 0 to disable downsampling.",
    )
    parser.add_argument(
        "--auto-refine",
        action="store_true",
        help="Run a second full-resolution pass for uncertain detections.",
    )
    parser.add_argument(
        "--refine-confidence",
        type=float,
        default=0.60,
        help="Confidence threshold below which detections are refined when --auto-refine is enabled.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    detector = TeapotHandleDetector(
        target_max_dim=args.target_max_dim,
        auto_refine=args.auto_refine,
        refine_confidence=args.refine_confidence,
    )

    images = detector.collect_images(input_path)
    if not images:
        print(f"No images found in: {input_path}")
        return

    print("image,handle_side,center_x,center_y,confidence,note")
    for image_path in images:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        result, mask, overlay = detector.detect_auto(
            rgb,
            image_name=image_path.name,
        )

        if output_dir is not None:
            detector.save_outputs(output_dir, image_path, rgb, overlay, mask)

        cx = "" if result.center_x is None else str(result.center_x)
        cy = "" if result.center_y is None else str(result.center_y)
        print(
            f"{result.image_name},{result.handle_side},{cx},{cy},{result.confidence:.2f},{result.note}"
        )


if __name__ == "__main__":
    main()
