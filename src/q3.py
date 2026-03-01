import cv2
import numpy as np

TURF_PATH = "assets/turf.jpg"
FLAG_PATH = "assets/flag.png"
OUTPUT_PATH = "output/turf_with_flag.jpg"

ALPHA = 0.50           # overall alpha blend; 0 = turf only, 1 = flag only
FEATHER = 31           # edge softness (odd number)
ROTATE_FLAG_K = 1      # 0=no rotate, 1=90° CW, 2=180, 3=270 CW

SAT_SCALE = 0.50       # <1 makes it less "neon"
VAL_SCALE = 0.80      # <1 slightly darker
TEXTURE_MIX = 1     # how much turf texture imprints on the flag [0..1]
GAUSS_BLUR = 5         # small blur on warped flag to blend into scene


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
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)  # less neon by reducing saturation; clip to valid range (just in case)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255) # slightly darker by reducing value; clip to valid range (just in case)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def overlay_flag_realistic(turf, flag, dst_quad,
                           alpha, feather,
                           texture_mix, gauss_blur):
    """ 
    Overlay the flag onto the turf with a perspective warp defined by dst_quad (4 corners in TL,TR,BR,BL order).
    The flag is blended using alpha blending, with optional feathering at the edges and modulation by the turf texture for realism.
    
    
    Args:
        turf: HxWx3 BGR image of the turf background.
        flag: HxWx3 BGR image of the flag to overlay.
        dst_quad: 4x2 array of (x,y) coordinates for the corners of the quadrilateral on the turf where the flag should be mapped. Order should be TL, TR, BR, BL.
        alpha: Overall blending factor for the flag (0=only turf, 1=only flag).
        feather: Size of Gaussian blur kernel for feathering edges (odd integer). Higher means softer edges.
        texture_mix: How much to imprint the turf texture onto the flag (0=no texture, 1=full texture).
        gauss_blur: Size of Gaussian blur kernel to apply to warped flag for better blending (odd integer).
    Returns:
        Blended image of the same size as turf with the flag overlaid.
    """
    Ht, Wt = turf.shape[:2]
    Hf, Wf = flag.shape[:2]

    src = np.array([[0, 0],
                    [Wf-1, 0],
                    [Wf-1, Hf-1],
                    [0, Hf-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst_quad) # 3x3 homography matrix to warp flag onto the destination quad
    warped_flag = cv2.warpPerspective(flag, H, (Wt, Ht), flags=cv2.INTER_LINEAR)    # interpolate for smoothness

    if gauss_blur and gauss_blur > 0:   # blur to make it more like painted on the surface
        k = gauss_blur if gauss_blur % 2 == 1 else gauss_blur + 1
        warped_flag = cv2.GaussianBlur(warped_flag, (k, k), 0)
    
    # apply a mask to blend only in the warped area of the flag
    white = np.ones((Hf, Wf), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(white, H, (Wt, Ht), flags=cv2.INTER_NEAREST)
    if feather and feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)  # feather the edges to make it natural
    m = (warped_mask.astype(np.float32) / 255.0)[:, :, None]    # normalize mask to [0..1] and make it 3-channel (for mixing with color images)

    # imprint turf texture onto flag (makes it look painted/printed)
    turf_gray = cv2.cvtColor(turf, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    turf_blur = cv2.GaussianBlur(turf_gray, (0, 0), 5)
    texture = np.clip((turf_gray - turf_blur), -0.25, 0.25)  # high-frequency component ~ grass texture
    warped_flag_f = warped_flag.astype(np.float32) / 255.0  # normalize to [0..1] for blending
    mod = (1.0 + texture_mix * texture)[:, :, None] # modulate value for brightness; make 3 channel
    warped_flag_f = np.clip(warped_flag_f * mod, 0, 1)

    # ---- final alpha blend only in masked region ----
    turf_f = turf.astype(np.float32) / 255.0
    out = turf_f * (1 - alpha*m) + warped_flag_f * (alpha*m)    # apply mask to make realistic.
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

    # Rotation of flag + desaturation
    flag = rotate_flag(flag, ROTATE_FLAG_K)
    flag = make_flag_less_saturated(flag, SAT_SCALE, VAL_SCALE)

    img_display = turf.copy()
    points = []

    cv2.namedWindow("Click 4 corners (R=reset, ESC=quit)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Click 4 corners (R=reset, ESC=quit)", 1200, 800)  # resizable window with default size
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

    # cv2.destroyAllWindows()

    dst_quad = order_points_by_angle(points)
    print("\nOrdered quad (TL,TR,BR,BL):\n", dst_quad)

    result = overlay_flag_realistic(
        turf, flag, dst_quad,
        alpha=ALPHA, feather=FEATHER,
        texture_mix=TEXTURE_MIX, gauss_blur=GAUSS_BLUR
    )

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Saved: {OUTPUT_PATH}")

    cv2.namedWindow("Result (realistic)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result (realistic)", 1200, 800)  # resizable output window
    cv2.imshow("Result (realistic)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()