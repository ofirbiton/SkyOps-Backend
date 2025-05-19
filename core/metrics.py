import numpy as np
import math
from core.graph_builder import line_intersects_building

# Returns the number of nodes in the path
def count_nodes(path):
    return len(path)

# Computes the total length of the path
def compute_path_length(path):
    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:])
    )

# Computes the average step length in the path
def compute_average_step_length(path):
    if len(path) < 2:
        return 0
    return compute_path_length(path) / (len(path) - 1)

# Computes the path density given the width and height of the area
def compute_path_density(path, width, height):
    area = width * height
    density = len(path) / area if area > 0 else 0
    print(f"  ➤ Path density: {density:.6f} nodes per pixel²")
    return density

# Compares the raw and optimized paths and prints metrics
def compare_paths(path_raw, path_opt):
    len_raw = count_nodes(path_raw)
    len_opt = count_nodes(path_opt)
    len_diff = len_raw - len_opt
    reduction_percent = (len_diff / len_raw) * 100 if len_raw else 0

    dist_raw = compute_path_length(path_raw)
    dist_opt = compute_path_length(path_opt)
    length_reduction_percent = (1 - dist_opt / dist_raw) * 100 if dist_raw else 0

    print("🔍 Fine-Tuning Metrics:")
    print(f"  ➤ Nodes before:  {len_raw}")
    print(f"  ➤ Nodes after:   {len_opt}")
    print(f"  ➤ Node reduction: {len_diff} ({reduction_percent:.2f}%)")
    print(f"  ➤ Path length before: {dist_raw:.2f} px")
    print(f"  ➤ Path length after:  {dist_opt:.2f} px")
    print(f"  ➤ Length change:       {dist_raw - dist_opt:.2f} px ({length_reduction_percent:.2f}%)")

# Prints the main path metrics: length, node count, detour ratio, and deviation from straight line
def print_all_metrics(path_raw, path_opt, takeoff, landing, building_mask, image_size):
    print("🔍 Metrics Evaluation:")
    dist_raw = compute_path_length(path_raw)
    dist_opt = compute_path_length(path_opt)
    print(f"  ➤ Path length before: {dist_raw:.2f} px")
    print(f"  ➤ Path length after:  {dist_opt:.2f} px")
    len_raw = count_nodes(path_raw)
    len_opt = count_nodes(path_opt)
    print(f"  ➤ Node count before: {len_raw}")
    print(f"  ➤ Node count after:  {len_opt}")
    ratio = detour_ratio(path_opt, takeoff, landing)
    print(f"  ➤ Detour ratio (path/straight): {ratio:.2f}x")
    avg_dev = deviation_from_straight_line(path_opt, takeoff, landing)
    print(f"  ➤ Avg. deviation from straight line: {avg_dev:.2f} px")
    print("=====================================")

# Computes the number of sharp turns in the path
def compute_angle_changes(path):
    def angle(a, b, c):
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    count = 0
    for i in range(1, len(path) - 1):
        ang = angle(path[i - 1], path[i], path[i + 1])
        if ang < 150:
            count += 1

    percent = (count / (len(path) - 2)) * 100 if len(path) >= 3 else 0
    print(f"  ➤ Sharp turns: {count} ({percent:.2f}%)")
    return count

# Computes the average deviation from the straight line between takeoff and landing
def deviation_from_straight_line(path, takeoff, landing):
    x0, y0 = takeoff
    x1, y1 = landing
    line_vec = np.array([x1 - x0, y1 - y0])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0

    deviation_sum = 0
    for (x, y) in path:
        point_vec = np.array([x - x0, y - y0])
        proj_len = np.dot(point_vec, line_vec) / line_len
        proj_point = (x0 + proj_len * line_vec[0] / line_len,
                      y0 + proj_len * line_vec[1] / line_len)
        deviation = np.linalg.norm(np.array([x, y]) - proj_point)
        deviation_sum += deviation

    avg_deviation = deviation_sum / len(path)
    percent = (avg_deviation / line_len) * 100
    return avg_deviation

# Computes the detour ratio (path length / straight line)
def detour_ratio(path, takeoff, landing):
    direct_dist = np.linalg.norm(np.array(takeoff) - np.array(landing))
    path_len = compute_path_length(path)
    ratio = path_len / direct_dist if direct_dist != 0 else 1
    return ratio

# Checks if the path intersects with any buildings
def check_for_building_collisions(path, building_mask):
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        if line_intersects_building(x1, y1, x2, y2, building_mask):
            print("  ➤ ⚠ Path crosses building!")
            return True
    print("  ➤ ✅ Path is clear of buildings.")
    return False

# Computes the maximum step length in the path
def max_step_length(path):
    if len(path) < 2:
        return 0
    steps = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(path[:-1], path[1:])]
    max_step = max(steps)
    print(f"  ➤ Max step length: {max_step:.2f} px")
    return max_step

# Computes the ratio of straight segments in the path
def straight_line_ratio(path, angle_threshold=30):
    def angle(a, b, c):
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    count_straight = 0
    for i in range(1, len(path) - 1):
        ang = angle(path[i - 1], path[i], path[i + 1])
        if ang > 180 - angle_threshold:
            count_straight += 1

    percent = (count_straight / (len(path) - 2)) * 100 if len(path) >= 3 else 0
    print(f"  ➤ Straight segments ratio: {count_straight} ({percent:.2f}%)")
    return percent

# Computes the average smoothness angle of the path
def path_smoothness(path):
    def angle(a, b, c):
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    angles = [angle(path[i - 1], path[i], path[i + 1]) for i in range(1, len(path) - 1)]
    if not angles:
        return 0
    smoothness = sum(angles) / len(angles)
    print(f"  ➤ Avg. smoothness angle: {smoothness:.2f}°")
    return smoothness