import cv2
import numpy as np

TURF_PATH = "assets/turf.jpg"
FLAG_PATH = "assets/flag.png"

ALPHA = 0.80           # overall alpha blend
FEATHER = 31           # edge softness (odd number)
ROTATE_FLAG_K = 1      # 0=no rotate, 1=90° CW, 2=180, 3=270 CW

SAT_SCALE = 0.85       # <1 makes it less "neon"
VAL_SCALE = 0.98       # <1 slightly darker
TEXTURE_MIX = 0.35     # how much turf texture imprints on the flag [0..1]
GAUSS_BLUR = 3         # small blur on warped flag to blend into scene


points = []
img_display = None


def mouse_callback(event, x, y, flags, param):
    """ Mouse callback to collect 4 corner points on the image. Left-click to add points, R to reset last point. """
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4 and img_display is not None:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(points)), (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Click 4 corners (R=reset, ESC=quit)", img_display)


def order_points_by_angle(pts):
    """ Order 4 points in TL, TR, BR, BL order by sorting based on angle from centroid."""
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    pts = pts[np.argsort(angles)]  # CCW; consider centroid as origin and sort by angle asc to get TL, TR, BR, BL in CCW order
    return pts


def rotate_flag(flag_bgr, k):
    """Rotate by 90° clockwise k times."""
    k = k % 4
    if k == 0:
        return flag_bgr
    return np.ascontiguousarray(np.rot90(flag_bgr, -k))  # -k = clockwise


def make_flag_less_saturated(flag_bgr, sat_scale=0.85, val_scale=0.98):
    hsv = cv2.cvtColor(flag_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 2] *= val_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def overlay_flag_realistic(turf, flag, dst_quad,
                           alpha=0.8, feather=31,
                           texture_mix=0.35, gauss_blur=3):
    Ht, Wt = turf.shape[:2]
    Hf, Wf = flag.shape[:2]

    src = np.array([[0, 0],
                    [Wf-1, 0],
                    [Wf-1, Hf-1],
                    [0, Hf-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst_quad)

    warped_flag = cv2.warpPerspective(flag, H, (Wt, Ht), flags=cv2.INTER_LINEAR)

    # mask by warping a white rectangle (guaranteed correct)
    white = np.ones((Hf, Wf), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(white, H, (Wt, Ht), flags=cv2.INTER_NEAREST)

    # feather the mask edges
    if feather and feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

    # small blur on flag helps it "sit" on the surface
    if gauss_blur and gauss_blur > 0:
        k = gauss_blur if gauss_blur % 2 == 1 else gauss_blur + 1
        warped_flag = cv2.GaussianBlur(warped_flag, (k, k), 0)

    m = (warped_mask.astype(np.float32) / 255.0)[:, :, None]

    # ---- imprint turf texture onto flag (makes it look painted/printed) ----
    # Use high-frequency detail of turf (edge/texture map) to modulate the flag brightness slightly
    turf_gray = cv2.cvtColor(turf, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    turf_blur = cv2.GaussianBlur(turf_gray, (0, 0), 5)
    texture = (turf_gray - turf_blur)  # high-frequency component ~ grass texture
    texture = np.clip(texture, -0.25, 0.25)

    warped_flag_f = warped_flag.astype(np.float32) / 255.0
    # modulate flag by (1 + texture_mix * texture)
    mod = (1.0 + texture_mix * texture)[:, :, None]
    warped_flag_f = np.clip(warped_flag_f * mod, 0, 1)

    # ---- final alpha blend only in masked region ----
    turf_f = turf.astype(np.float32) / 255.0
    out = turf_f * (1 - alpha*m) + warped_flag_f * (alpha*m)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return out


def main():
    global img_display, points

    turf = cv2.imread(TURF_PATH)
    if turf is None:
        raise FileNotFoundError(f"Could not read {TURF_PATH}")

    flag = cv2.imread(FLAG_PATH)
    if flag is None:
        raise FileNotFoundError(f"Could not read {FLAG_PATH}")

    # Apply rotation + color realism before warping
    flag = rotate_flag(flag, ROTATE_FLAG_K)
    flag = make_flag_less_saturated(flag, SAT_SCALE, VAL_SCALE)

    img_display = turf.copy()
    points = []

    cv2.namedWindow("Click 4 corners (R=reset, ESC=quit)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click 4 corners (R=reset, ESC=quit)", mouse_callback)

    while True:
        cv2.imshow("Click 4 corners (R=reset, ESC=quit)", img_display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return
        if key in (ord('r'), ord('R')):
            points = []
            img_display = turf.copy()
            print("Reset points.")
        if len(points) == 4:
            break

    cv2.destroyAllWindows()

    dst_quad = order_points_by_angle(points)
    print("\nOrdered quad (TL,TR,BR,BL):\n", dst_quad)

    result = overlay_flag_realistic(
        turf, flag, dst_quad,
        alpha=ALPHA, feather=FEATHER,
        texture_mix=TEXTURE_MIX, gauss_blur=GAUSS_BLUR
    )

    cv2.imwrite("turf_with_flag_realistic.jpg", result)
    print("Saved: turf_with_flag_realistic.jpg")

    cv2.imshow("Result (realistic)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()