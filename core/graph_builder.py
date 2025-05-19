# graphBuilder.py

from shared import dependencies as dep
import math

# Returns all (x, y) points on the line between (x1, y1) and (x2, y2) using Bresenham's algorithm
def bresenham_line_points(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    cx, cy = x1, y1
    while True:
        points.append((cx, cy))
        if cx == x2 and cy == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            cx += sx
        if e2 <= dx:
            err += dx
            cy += sy
    return points

# Returns True if the line between (x1, y1) and (x2, y2) does not cross any building pixels in merged_image
def is_line_clear_of_buildings(merged_image, x1, y1, x2, y2):
    line_pixels = bresenham_line_points(x1, y1, x2, y2)
    for px, py in line_pixels:
        if py < 0 or px < 0 or py >= merged_image.shape[0] or px >= merged_image.shape[1]:
            return False
        if (merged_image[py, px] == [255, 255, 255]).all():
            return False
    return True

# Finds all yellow junctions reachable from (y, x) via skeleton pixels using BFS
def find_neighbors(y, x, skeleton_mask, yellow_mask):
    neighbors = []
    queue = dep.deque([(y, x)])
    visited = {(y, x)}
    while queue:
        cy, cx = queue.popleft()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in visited:
                    continue
                if ny < 0 or nx < 0 or ny >= skeleton_mask.shape[0] or nx >= skeleton_mask.shape[1]:
                    continue
                if skeleton_mask[ny, nx] > 0:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
                if yellow_mask[ny, nx] > 0 and (ny, nx) != (y, x):
                    neighbors.append((ny, nx))
    return neighbors

# Connects yellow junctions via skeleton, returns image with red lines, node list, and adjacency dict
def connect_yellow_junctions(merged_image, yellow_mask, skeleton_mask):
    image_with_lines = merged_image.copy()
    yellow_coords = dep.np.column_stack(dep.np.where(yellow_mask > 0))
    node_list = [(x, y) for (y, x) in yellow_coords]
    adjacency_dict = {node: [] for node in node_list}
    for (x, y) in node_list:
        dep.cv2.rectangle(image_with_lines, (x - 1, y - 1), (x + 1, y + 1), (255, 255, 0), -1)
    connected_pairs = set()
    for (x1, y1) in node_list:
        neighbors_list = find_neighbors(y1, x1, skeleton_mask, yellow_mask)
        for (ny, nx) in neighbors_list:
            (x2, y2) = (nx, ny)
            if ((x1, y1), (x2, y2)) in connected_pairs or ((x2, y2), (x1, y1)) in connected_pairs:
                continue
            connected_pairs.add(((x1, y1), (x2, y2)))
            if not is_line_clear_of_buildings(image_with_lines, x1, y1, x2, y2):
                continue
            dist = float(dep.np.hypot(x2 - x1, y2 - y1))
            adjacency_dict[(x1, y1)].append([(x1, y1), (x2, y2), dist])
            adjacency_dict[(x2, y2)].append([(x2, y2), (x1, y1), dist])
            start_x = x1 + dep.np.sign(x2 - x1)
            start_y = y1 + dep.np.sign(y2 - y1)
            end_x = x2 - dep.np.sign(x2 - x1)
            end_y = y2 - dep.np.sign(y2 - y1)
            dep.cv2.line(image_with_lines, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1)
    return image_with_lines, node_list, adjacency_dict

# Adds new_xy as a node to adjacency_dict and connects to nearest node if possible
def add_point_to_graph(new_xy, adjacency_dict, building_mask, image_to_draw, ignore_building=False):
    (new_x, new_y) = new_xy
    existing_nodes = list(adjacency_dict.keys())
    if not existing_nodes:
        return [new_xy]
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    existing_nodes_sorted = sorted(existing_nodes, key=lambda node: dist(new_xy, node))
    for candidate_node in existing_nodes_sorted:
        x2, y2 = candidate_node
        if ignore_building or not line_intersects_building(new_x, new_y, x2, y2, building_mask):
            distance_val = dist(new_xy, candidate_node)
            if new_xy not in adjacency_dict:
                adjacency_dict[new_xy] = []
            adjacency_dict[new_xy].append([new_xy, candidate_node, distance_val])
            adjacency_dict[candidate_node].append([candidate_node, new_xy, distance_val])
            dep.cv2.circle(image_to_draw, new_xy, 3, (0, 255, 255), -1)
            dep.cv2.line(image_to_draw, new_xy, candidate_node, (0, 0, 255), 2)
            return new_xy
    return None

# Returns True if the line between (x1, y1) and (x2, y2) crosses a building in building_mask
def line_intersects_building(x1, y1, x2, y2, building_mask):
    points_on_line = bresenham_line_points(x1, y1, x2, y2)
    for (px, py) in points_on_line:
        if py < 0 or py >= building_mask.shape[0] or px < 0 or px >= building_mask.shape[1]:
            return True
        if building_mask[py, px] == 1:
            return True
    return False