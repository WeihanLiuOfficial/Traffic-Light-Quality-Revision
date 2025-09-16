import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def detect_lights(image_path, brightness_ratio=0.2, min_radius=20, max_radius=60, min_bright_fraction=0.8):
    """
    Robust LED detection:
    - include brightest point
    - find bright regions
    - split merged close LEDs
    - fit circles
    - only keep circles where most pixels are bright
    - combine superposed circles
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find brightest pixel
    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    brightest_x, brightest_y = maxLoc
    brightest_color = img[brightest_y, brightest_x].tolist()  # BGR

    # Threshold bright regions
    thresh_val = int(brightness_ratio * maxVal)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    lights = []
    for i in range(1, num_labels):  # skip background
        # Component mask
        mask_comp = (labels == i).astype(np.uint8) * 255
        x_c, y_c = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        radius = int(np.sqrt(area / np.pi))  # approximate radius

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
                        lights.append(Light(x, y, int(r * 0.7), brightest_color))
                continue  # skip adding merged component itself

        # Otherwise, treat as single light
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(x_c), int(y_c)), radius, 255, -1)
        bright_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        total_pixels = np.count_nonzero(mask)
        bright_fraction = bright_pixels / total_pixels
        if bright_fraction >= min_bright_fraction:
            lights.append(Light(int(x_c), int(y_c), radius, brightest_color))

    # ----------------------
    # Merge overlapping circles
    merged = []
    while lights:
        base = lights.pop(0)
        overlapping = [base]
        rest = []
        for other in lights:
            dist = np.hypot(base.x - other.x, base.y - other.y)
            if dist < (base.radius + other.radius) * 0.7:  # overlap threshold
                overlapping.append(other)
            else:
                rest.append(other)
        # Merge all overlapping circles
        if len(overlapping) == 1:
            merged.append(base)
        else:
            # Weighted average center
            sum_x = sum(c.x * c.radius for c in overlapping)
            sum_y = sum(c.y * c.radius for c in overlapping)
            sum_r = sum(c.radius for c in overlapping)
            new_x = int(sum_x / sum_r)
            new_y = int(sum_y / sum_r)
            # New radius covers all
            max_dist = max(np.hypot(c.x - new_x, c.y - new_y) + c.radius for c in overlapping)
            merged.append(Light(new_x, new_y, int(max_dist), brightest_color))
        lights = rest

    return merged



def compare_boards(golden_lights_lst, test_lights, img, pos_tolerance=60, color_tolerance=50):
    """
    Compares two light boards by position and color.
    - Marks matched lights in GREEN
    - Marks unmatched golden lights in RED
    """

    matches = []
    golden_copy = golden_lights_lst.copy()  # to remove matched golden lights

    # Loop over test lights
    for t in test_lights:
        matched_index = None
        for i, g in enumerate(golden_copy):
            dist = np.hypot(g.x - t.x, g.y - t.y)
            color_diff = np.linalg.norm(np.array(g.color) - np.array(t.color))
            if dist <= pos_tolerance and color_diff <= color_tolerance:
                matched_index = i
                break  # stop at first matching golden light

        if matched_index is not None:
            matches.append(t)
            golden_copy.pop(matched_index)  # remove matched golden light

    # Draw matched lights in GREEN
    for m in matches:
        cv2.circle(img, (m.x, m.y), m.radius, (0, 255, 0), 2)

    # Draw remaining unmatched golden lights in RED
    average_test_radius = int(np.mean([t.radius for t in test_lights]))
    for g in golden_copy:
        cv2.circle(img, (g.x, g.y), average_test_radius, (0, 0, 255), 2)

    # Board passes if no golden lights left unmatched
    board_passed = (len(golden_copy) == 0)
    return board_passed, img


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
    golden_img = "gs3.jpg"
    test_img = "test3changed.jpg"
    #test_lights = show_detected_lights(test_img)
    result = run_testing_board(golden_img, test_img)
    print(result)
