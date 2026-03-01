import numpy as np
import cv2
import matplotlib.pyplot as plt

FOCAL_LENGTH_MM = 8.0          # f
OBJECT_DISTANCE_MM = 720.0     # NOTE: This is wrong in the question. D (lens to object plane)
PIXEL_SIZE_UM = 2.2            # square pixels: 2.2um x 2.2um
PIXEL_SIZE_MM = PIXEL_SIZE_UM * 1e-3  # pixel size in mm


def pixel_to_sensor_mm(px: float) -> float:
    """Convert pixel length to physical length on sensor (mm)."""
    return px * PIXEL_SIZE_MM


def image_plane_distance_mm() -> float:
    """Compute D' from thin lens formula: 1/f = 1/D + 1/D'."""
    if OBJECT_DISTANCE_MM <= FOCAL_LENGTH_MM:
        raise ValueError("OBJECT_DISTANCE_MM must be greater than FOCAL_LENGTH_MM")
    return (FOCAL_LENGTH_MM * OBJECT_DISTANCE_MM) / (OBJECT_DISTANCE_MM - FOCAL_LENGTH_MM)


def sensor_to_object_mm(sensor_mm: float) -> float:
    """
    Convert sensor size y' (mm) to object size y (mm).

    Using similar triangles: y / y' = D / D'
    Using thin lens: 1/f = 1/D + 1/D'  => D' = fD / (D - f)
    """
    d_prime_mm = image_plane_distance_mm()
    return (OBJECT_DISTANCE_MM / d_prime_mm) * sensor_mm


def segment_earrings(bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of earring pixels.
    Strategy:
    - Segment non-white / "gold-ish" using HSV (higher saturation, not too bright)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Mask: not-white-ish pixels
    # - saturation > ~20 (out of 255) avoids pure whites/greys
    # - value < ~250 avoids very bright background
    mask = ((S > 20) & (V < 250)).astype(np.uint8) * 255

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_two_earrings(mask: np.ndarray):
    """
    Find contours and return the two largest components' bounding boxes.
    Returns list of dicts with contour, bbox, area.
    bbox = (x, y, w, h)
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    comps = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:  # ignore tiny specks
            continue
        x, y, w, h = cv2.boundingRect(c)
        comps.append({"contour": c, "area": area, "bbox": (x, y, w, h)})

    comps.sort(key=lambda d: d["area"], reverse=True)
    return comps[:2]    # largest = earring 1, second largest = earring 2 


def measure_and_print(image_path: str, show_plot: bool = True):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    mask = segment_earrings(bgr)
    earrings = find_two_earrings(mask)

    if len(earrings) < 2:
        print("Warning: Could not confidently find two earrings. "
              "Try adjusting HSV thresholds / crop ratio in segment_earrings().")

    d_prime_mm = image_plane_distance_mm()

    print("\n=== Measurements (bounding box based) ===")
    print(
        f"f = {FOCAL_LENGTH_MM} mm, D = {OBJECT_DISTANCE_MM} mm, "
        f"D' = {d_prime_mm:.4f} mm, pixel = {PIXEL_SIZE_UM} um"
    )

    results = []
    for i, e in enumerate(earrings, start=1):
        x, y, w, h = e["bbox"]

        # Pixels in sensor -> sensor mm
        w_sensor_mm = pixel_to_sensor_mm(w)
        h_sensor_mm = pixel_to_sensor_mm(h)

        # Sensor in mm -> object mm
        w_obj_mm = sensor_to_object_mm(w_sensor_mm)
        h_obj_mm = sensor_to_object_mm(h_sensor_mm)

        results.append((w_obj_mm, h_obj_mm))

        print(f"\nEarring {i}:")
        print(f"  bbox pixels (w,h) = ({w}, {h})")
        print(f"  sensor size (mm)  = ({w_sensor_mm:.4f}, {h_sensor_mm:.4f})")
        print(f"  object size (mm)  = ({w_obj_mm:.2f}, {h_obj_mm:.2f})")

    if len(results) == 2:
        avg_w = (results[0][0] + results[1][0]) / 2
        avg_h = (results[0][1] + results[1][1]) / 2
        avg_diameter = np.sqrt(avg_w**2 + avg_h**2)
        print("\nAverage earring size (mm) from the two boxes:")
        print(f"  avg width  = {avg_w:.2f} mm")
        print(f"  avg height = {avg_h:.2f} mm")
        print(f"  avg diameter = {avg_diameter:.2f} mm")


    # Visualization
    if show_plot:
        vis = bgr.copy()
        for i, e in enumerate(earrings, start=1):
            x, y, w, h = e["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"E{i}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show side-by-side: original + mask + annotated
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("Original")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Segmented mask")
        ax[1].axis("off")

        ax[2].imshow(vis_rgb)
        ax[2].set_title("Detected earrings (bbox)")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    measure_and_print("assets/earrings.jpg", show_plot=True)