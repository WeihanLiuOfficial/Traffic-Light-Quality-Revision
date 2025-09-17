import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


from hypothesis.core import running_under_pytest


class Light:
    """
    Class Light
    """
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color


def detect_lights(image_path, brightness_ratio=0.6, min_radius=20, max_radius=60, min_bright_fraction=0.8, resize_factor=0.3):
    """
    Robust LED detection with optional downscaling for speed:
    - include brightest point
    - find bright regions
    - split merged close LEDs with HoughCircles
    - fit circles
    - only keep circles where most pixels are bright
    - combine superposed circles
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    min_radius = int(min_radius * resize_factor)
    max_radius = int(max_radius * resize_factor)

    # Resize image
    if resize_factor != 1.0:
        img_small = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # Find brightest pixel
    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    brightest_x, brightest_y = maxLoc
    brightest_color = img_small[brightest_y, brightest_x].tolist()  # BGR

    # Threshold bright regions
    thresh_val = int(brightness_ratio * maxVal)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    lights = []
    for i in range(1, num_labels):  # skip background
        mask_comp = (labels == i).astype(np.uint8) * 255
        x_c, y_c = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        radius = int(np.sqrt(area / np.pi))

        if radius < min_radius or radius > max_radius:
            continue

        # Split large or merged components using HoughCircles
        if radius > 1.5 * min_radius:
            blurred = cv2.GaussianBlur(mask_comp, (5, 5), 0)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_radius,
                param1=50,
                param2=10,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for x, y, r in circles:
                    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    bright_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=circle_mask))
                    total_pixels = np.count_nonzero(circle_mask)
                    bright_fraction = bright_pixels / total_pixels
                    if bright_fraction >= min_bright_fraction:
                        # Scale coordinates and radius back to original image
                        orig_x = int(x / resize_factor)
                        orig_y = int(y / resize_factor)
                        orig_r = int(r / resize_factor)
                        lights.append(Light(orig_x, orig_y, orig_r, brightest_color))
                continue  # skip adding merged component itself

        # Otherwise, treat as single light
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(x_c), int(y_c)), radius, 255, -1)
        bright_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        total_pixels = np.count_nonzero(mask)
        bright_fraction = bright_pixels / total_pixels
        if bright_fraction >= min_bright_fraction:
            orig_x = int(x_c / resize_factor)
            orig_y = int(y_c / resize_factor)
            orig_radius = int(radius / resize_factor)
            lights.append(Light(orig_x, orig_y, orig_radius, brightest_color))

    # ----------------------
    # Merge overlapping circles
    if not lights:
        return []

        # Convert to arrays for vectorized operations
    xs = np.array([l.x for l in lights])
    ys = np.array([l.y for l in lights])
    rs = np.array([l.radius for l in lights])
    colors = [l.color for l in lights]

    merged = []
    used = np.zeros(len(lights), dtype=bool)

    for i in range(len(lights)):
        if used[i]:
            continue

        # Compute distance to all other lights
        dx = xs - xs[i]
        dy = ys - ys[i]
        dists = np.hypot(dx, dy)

        # Find overlapping lights
        overlap_mask = dists < (rs[i] + rs) * 0.7
        indices = np.where(overlap_mask & ~used)[0]

        # Merge them
        if len(indices) == 1:
            merged.append(lights[i])
        else:
            sum_r = rs[indices].sum()
            new_x = int(np.sum(xs[indices] * rs[indices]) / sum_r)
            new_y = int(np.sum(ys[indices] * rs[indices]) / sum_r)
            max_dist = max(np.hypot(xs[j] - new_x, ys[j] - new_y) + rs[j] for j in indices)
            merged.append(Light(new_x, new_y, int(max_dist), colors[i]))

        used[indices] = True

    return merged


def compare_boards(golden_lights, test_lights, img, pos_tolerance=60, color_tolerance=50):
    import numpy as np
    from scipy.spatial import cKDTree

    if not test_lights or not golden_lights:
        return False, img

    # Precompute positions and colors
    golden_positions = np.array([(g.x, g.y) for g in golden_lights])
    golden_colors = np.array([g.color for g in golden_lights])
    test_positions = np.array([(t.x, t.y) for t in test_lights])
    test_colors = np.array([t.color for t in test_lights])

    # Build KDTree for golden positions
    tree = cKDTree(golden_positions)

    # Query all test lights at once
    overlaps = tree.query_ball_point(test_positions, r=pos_tolerance)

    matches = []
    unmatched = set(range(len(golden_lights)))

    for i, candidates in enumerate(overlaps):
        if not candidates:
            continue
        t_color = test_colors[i]
        g_colors = golden_colors[candidates]
        color_diffs = np.linalg.norm(g_colors - t_color, axis=1)
        idx_match = np.where(color_diffs <= color_tolerance)[0]
        if idx_match.size > 0:
            golden_idx = candidates[idx_match[0]]
            matches.append(test_lights[i])
            unmatched.discard(golden_idx)

    # Draw matched lights in GREEN
    for m in matches:
        cv2.circle(img, (m.x, m.y), m.radius, (0, 255, 0), 2)

    # Compute average radius for unmatched golden lights
    avg_radius = int(np.mean([t.radius for t in test_lights])) if test_lights else 20

    # Draw unmatched golden lights in RED
    for i in unmatched:
        g = golden_lights[i]
        cv2.circle(img, (g.x, g.y), avg_radius, (0, 0, 255), 2)

    # Board passes if no golden lights left unmatched
    return len(unmatched) == 0, img


def run_testing_board(golden_img_path, test_img_path):
    """
    Tests board detection
    """
    # detect_lights returns only the lights list
    golden_lights = detect_lights(golden_img_path)
    test_lights = detect_lights(test_img_path)

    print(f"Golden board lights: {len(golden_lights)}")
    print(f"Test board lights:   {len(test_lights)}")

    # Load test image separately for drawing
    img = cv2.imread(test_img_path)
    passed, marked_img = compare_boards(golden_lights, test_lights, img, color_tolerance=80)

    if passed:
        result = "Board passes the test ✅"
    else:
        result = "Board doesn’t pass the test ❌"

    # Show results with matplotlib
    plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    plt.title("Result")
    plt.axis("off")
    plt.show()

    return result


def show_detected_lights(image_path):
    """
    Run detect_lights on an image and display the results.
    """
    lights = detect_lights(image_path)
    img = cv2.imread(image_path)

    # Draw circles for each detected light
    for light in lights:
        cv2.circle(img, (light.x, light.y), light.radius, (0, 255, 0), 2)
        cv2.circle(img, (light.x, light.y), 2, (0, 0, 255), 3)  # center point

    # Show with matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(image_path)
    plt.axis("off")
    plt.show()

    print(f"Detected {len(lights)} lights in sample board.")
    return lights


if __name__ == "__main__":
    golden_img = "golden_sample2.jpg"
    test_img = "test2changed.jpg"
    #test_lights = show_detected_lights(golden_img)
    result = run_testing_board(golden_img, test_img)
    print(result)
