import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def fit_line_tls(
        x: np.ndarray[tuple[Any], np.dtype[np.float64]], 
        y: np.ndarray[tuple[Any], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
    pts = np.column_stack([x, y])
    mu = pts.mean(axis=0)
    centered = pts - mu
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    a, b = normal / np.hypot(*normal)
    c = -(a * mu[0] + b * mu[1])
    return np.array([a, b, c])

def line_from_two_points(
        p1: np.ndarray[tuple[Any, Any], np.dtype[np.float64]], 
        p2: np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    return np.array([a, b, c]) / np.hypot(a, b)

def point_line_dist(
        line: np.ndarray[tuple[Any], np.dtype[np.float64]], 
        pts: np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[Any], np.dtype[np.float64]]:
    a, b, c = line
    return np.abs(a*pts[:,0] + b*pts[:,1] + c)

def abc_to_slope(
        line: np.ndarray[tuple[Any], np.dtype[np.float64]]
    ) -> str:
    a, b, c = line
    if abs(b) > 1e-12:
        m = -a / b
        k = -c / b
        return f"y = {m:.6f}x + {k:.6f}"
    else:
        return f"x = {-c/a:.6f}"

def ransac_fit_line(
        points: np.ndarray[tuple[Any, Any], np.dtype[np.float64]], 
        n_iters=4000, threshold=0.25, min_inliers=35, seed=0
    ) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    best_line = None
    best_inliers = None
    best_count = -1

    for _ in range(n_iters):
        i, j = rng.choice(len(points), 2, replace=False)
        line = line_from_two_points(points[i], points[j])
        d = point_line_dist(line, points)
        inliers = d < threshold
        count = inliers.sum()

        if count > best_count and count >= min_inliers:
            best_count = count
            best_line = line
            best_inliers = inliers

    # refine using TLS
    inlier_pts = points[best_inliers]
    refined = fit_line_tls(inlier_pts[:,0], inlier_pts[:,1])
    return refined, best_inliers

def fit_three_lines(
        points: np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    remaining = np.arange(len(points))
    lines = []
    masks = []

    for k in range(3):
        pts_rem = points[remaining]
        line, inliers = ransac_fit_line(pts_rem, seed=42+k)
        if inliers is None:
            break
        mask_full = np.zeros(len(points), dtype=bool)
        mask_full[remaining[inliers]] = True
        lines.append(line)
        masks.append(mask_full)
        remaining = remaining[~inliers]

    return lines, masks

# === Data ===
D = np.genfromtxt("data/lines.csv", delimiter=",", skip_header=1)   # same as read_csv but better
X_cols = D[:, :3]   # 2D array of shape (N,3) for x-coordinates of 3 lines
Y_cols = D[:, 3:]

# === TLS for first line ===
x1 = X_cols[:,0]    # get only first col 
y1 = Y_cols[:,0]
line_tls = fit_line_tls(x1, y1)

print(f"\n{'='*30}")
print("PART (a) - TLS (First Line)")
print(f"{'='*30}")
print("ax + by + c = 0 form:", line_tls)
print("Slope form:", abc_to_slope(line_tls))

pts1 = np.column_stack([x1, y1])

# === RANSAC for three lines ===

points_all = np.column_stack([X_cols.flatten(), Y_cols.flatten()])  # flattening because we want to treat all points together for RANSAC
lines, masks = fit_three_lines(points_all)

print(f"\n{'='*30}")
print("PART (b) - RANSAC (Three Lines)")
print(f"{'='*30}")

for i, L in enumerate(lines):
    print(f"\nLine {i+1}")
    print("ax + by + c = 0 form:", L)
    print("Slope form:", abc_to_slope(L))

# === Plotting === 
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# ---- Left: TLS ----
axes[0].scatter(pts1[:,0], pts1[:,1])
xs = np.linspace(pts1[:,0].min()-1, pts1[:,0].max()+1, 200)
ys = (-line_tls[0]*xs - line_tls[2]) / line_tls[1]
axes[0].plot(xs, ys)
axes[0].set_title("(a) Total Least Squares - First Line")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].axis("equal")
axes[0].grid(True)

# ---- Right: RANSAC ----
remaining = np.ones(len(points_all), dtype=bool)
colors = ["blue","orange","green"]

for i, m in enumerate(masks):
    remaining &= ~m
    axes[1].scatter(points_all[m,0], points_all[m,1], color=colors[i])

if remaining.any():
    axes[1].scatter(points_all[remaining,0], points_all[remaining,1], color="red")

xs = np.linspace(points_all[:,0].min()-1, points_all[:,0].max()+1, 200)
for L in lines:
    if abs(L[1]) > 1e-12:
        ys = (-L[0]*xs - L[2]) / L[1]
        axes[1].plot(xs, ys)

axes[1].set_title("(b) RANSAC - Three Lines")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].axis("equal")
axes[1].grid(True)

plt.tight_layout()
plt.show()