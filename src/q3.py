import cv2
import numpy as np

TURF_PATH = "assets/turf.jpg"
FLAG_PATH = "assets/flag.png"
ALPHA = 0.85   # transparency

points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img_display, str(len(points)), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Click 4 corners (R=reset, ESC=quit)", img_display)

def order_points_by_angle(pts):
    """
    Robust ordering:
    - Sort points by angle around centroid (gives CW or CCW order)
    - Rotate so that first point is the top-left (min x+y)
    Returns points in clockwise order: TL, TR, BR, BL
    """
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)

    angles = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    idx = np.argsort(angles)  # CCW order
    pts = pts[idx]

    # rotate so top-left is first (min x+y)
    s = pts[:,0] + pts[:,1]
    start = np.argmin(s)
    pts = np.roll(pts, -start, axis=0)

    # ensure clockwise (OpenCV doesn't strictly require, but it's nice)
    # compute signed area: if positive -> CCW, reverse to CW
    area2 = 0
    for i in range(4):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1)%4]
        area2 += (x1*y2 - x2*y1)
    if area2 > 0:  # CCW
        pts = pts[[0,3,2,1]]

    return pts

def overlay_flag(turf, flag, dst_quad, alpha=0.85, feather=25):
    Ht, Wt = turf.shape[:2]
    Hf, Wf = flag.shape[:2]

    src = np.array([[0, 0],
                    [Wf-1, 0],
                    [Wf-1, Hf-1],
                    [0, Hf-1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst_quad)

    # Warp the flag
    warped_flag = cv2.warpPerspective(flag, H, (Wt, Ht), flags=cv2.INTER_LINEAR)

    # BEST MASK: warp a white rectangle using the SAME homography (guaranteed match)
    white = np.ones((Hf, Wf), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(white, H, (Wt, Ht), flags=cv2.INTER_NEAREST)

    # Feather edge for blending realism
    if feather and feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)

    m = (warped_mask.astype(np.float32) / 255.0)[:, :, None]

    out = turf.astype(np.float32) * (1 - alpha*m) + warped_flag.astype(np.float32) * (alpha*m)
    return np.clip(out, 0, 255).astype(np.uint8), warped_flag, warped_mask

def main():
    global img_display, points

    turf = cv2.imread(TURF_PATH)
    if turf is None:
        raise FileNotFoundError(f"Could not read {TURF_PATH}")

    flag = cv2.imread(FLAG_PATH)
    if flag is None:
        raise FileNotFoundError(f"Could not read {FLAG_PATH}")

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

    # robust ordering
    dst_quad = order_points_by_angle(points)
    print("\nOrdered quad (TL,TR,BR,BL):\n", dst_quad)

    result, warped_flag, warped_mask = overlay_flag(turf, flag, dst_quad, alpha=ALPHA)

    # Debug windows (so you can SEE what's happening)
    cv2.imshow("Warped Flag (debug)", warped_flag)
    cv2.imshow("Warped Mask (debug)", warped_mask)
    cv2.imshow("Result", result)
    cv2.imwrite("turf_with_flag.jpg", result)
    print("Saved: turf_with_flag.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()