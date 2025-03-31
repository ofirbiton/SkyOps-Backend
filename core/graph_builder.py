# graphBuilder.py

from shared import dependencies as dep
import math

def bresenham_line_points(x1, y1, x2, y2):
    """
    מחזירה את כל נקודות הקו (x,y) בין (x1, y1) ל-(x2, y2) לפי אלגוריתם Bresenham.
    """
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


def is_line_clear_of_buildings(merged_image, x1, y1, x2, y2):
    """
    בודק אם הקו בין (x1, y1) ל-(x2, y2) עובר רק ברקע (לא חוצה מבנים).
    מניח שהמבנה הוא פיקסל לבן [255, 255, 255].
    
    merged_image[y, x] -> [B, G, R]
    אם merged_image[py, px] == [255, 255, 255], נחשב מבנה.
    """
    line_pixels = bresenham_line_points(x1, y1, x2, y2)
    for px, py in line_pixels:
        # נוודא שהפיקסל בממדים
        if py < 0 or px < 0 or py >= merged_image.shape[0] or px >= merged_image.shape[1]:
            # אם יוצא מהתמונה, נחשיב כאילו יש מבנה (או נתעלם, לבחירתך)
            return False
        if (merged_image[py, px] == [255, 255, 255]).all():
            return False
    return True


def find_neighbors(y, x, skeleton_mask, yellow_mask):
    """
    מוצא את כל הצמתים הצהובים (yellow junctions) שאליהם ניתן להגיע
    מהצומת (y, x) דרך פיקסלים של השלד (skeleton_mask).
    מחזיר רשימה של (ny, nx) עבור כל צומת צהוב שמתגלה בשיטוט BFS.
    """
    neighbors = []
    queue = dep.deque([(y, x)])
    visited = {(y, x)}

    while queue:
        cy, cx = queue.popleft()
        # עובר על הסביבה [(-1, -1), ..., (1, 1)]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in visited:
                    continue
                if ny < 0 or nx < 0 or ny >= skeleton_mask.shape[0] or nx >= skeleton_mask.shape[1]:
                    continue

                # אם זהו פיקסל כחול/שלד
                if skeleton_mask[ny, nx] > 0:
                    visited.add((ny, nx))
                    queue.append((ny, nx))

                # אם זהו פיקסל צהוב (צומת) – נשמור אותו
                if yellow_mask[ny, nx] > 0 and (ny, nx) != (y, x):
                    neighbors.append((ny, nx))
    return neighbors


def connect_yellow_junctions(merged_image, yellow_mask, skeleton_mask):
    """
    עובר על כל הצמתים הצהובים ובודק מי שכן שלהם דרך השלד.
    אם הקו בין שני צמתים אינו חוצה מבנים (פיקסלים לבנים ב-merged_image),
    נוסיף קשת ל-adjacency_dict ונסמן קו אדום על image_with_lines.

    מחזיר:
      image_with_lines - עותק של merged_image עם קווים אדומים
      node_list - רשימת הצמתים (x, y)
      adjacency_dict - מילון: מפתח הוא (x, y), ערך הוא רשימת קשתות [[(x1, y1),(x2, y2), dist], ...]
    """
    image_with_lines = merged_image.copy()

    # מוציאים את כל הקואורדינטות הצהובות
    yellow_coords = dep.np.column_stack(dep.np.where(yellow_mask > 0))  # מחזיר array של (y, x)
    node_list = [(x, y) for (y, x) in yellow_coords]

    # מכינים adjacency_dict ריק
    adjacency_dict = {node: [] for node in node_list}

    # נסמן כל צומת צהוב קטן בתמונה להמחשה
    for (x, y) in node_list:
        dep.cv2.rectangle(image_with_lines, (x - 1, y - 1), (x + 1, y + 1), (255, 255, 0), -1)

    connected_pairs = set()

    # עוברים על כל הצמתים
    for (x1, y1) in node_list:
        # מוצאים צמתים צהובים שכנים לפי השלד
        neighbors_list = find_neighbors(y1, x1, skeleton_mask, yellow_mask)

        # עבור כל שכן, נבדוק אם ניתן לקשר בקו ישיר ללא חציית מבנה
        for (ny, nx) in neighbors_list:
            (x2, y2) = (nx, ny)

            # נבדוק אם לא טיפלנו כבר בצמד זה
            if ((x1, y1), (x2, y2)) in connected_pairs or ((x2, y2), (x1, y1)) in connected_pairs:
                continue
            connected_pairs.add(((x1, y1), (x2, y2)))

            # בודק אם הקו בין הצמתים לא עובר על מבנה
            if not is_line_clear_of_buildings(image_with_lines, x1, y1, x2, y2):
                continue

            dist = float(dep.np.hypot(x2 - x1, y2 - y1))
            adjacency_dict[(x1, y1)].append([(x1, y1), (x2, y2), dist])
            adjacency_dict[(x2, y2)].append([(x2, y2), (x1, y1), dist])

            # מצייר קו אדום
            start_x = x1 + dep.np.sign(x2 - x1)
            start_y = y1 + dep.np.sign(y2 - y1)
            end_x   = x2 - dep.np.sign(x2 - x1)
            end_y   = y2 - dep.np.sign(y2 - y1)
            dep.cv2.line(image_with_lines, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)

    return image_with_lines, node_list, adjacency_dict

import math

def add_point_to_graph(new_xy, adjacency_dict, building_mask):
    """
    מוסיף את הנקודה new_xy = (x, y) כצומת חדש ב-adjacency_dict,
    ומקשר אותה לצומת הקיים (existing_node) הקרוב ביותר שהקו אליו לא חוצה מבנה.
    אם לא מוצא אף צומת כזה => מחזיר None.
    אחרת מחזיר new_xy.
    """
    (new_x, new_y) = new_xy

    existing_nodes = list(adjacency_dict.keys())
    if not existing_nodes:
        return None  # אין למי להתחבר

    # מיין לפי מרחק
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    existing_nodes_sorted = sorted(
        existing_nodes,
        key=lambda node: dist(new_xy, node)
    )

    for candidate_node in existing_nodes_sorted:
        x2, y2 = candidate_node
        # אם הקו חוצה מבנה => לא טוב
        if not line_intersects_building(new_x, new_y, x2, y2, building_mask):
            # מעולה, נוכל לחבר
            distance_val = dist(new_xy, candidate_node)
            # 1) אם new_xy לא קיים עדיין ב-adjacency_dict => נוסיף
            if new_xy not in adjacency_dict:
                adjacency_dict[new_xy] = []

            # 2) הוסף קשת דו כיוונית
            adjacency_dict[new_xy].append((candidate_node, distance_val))
            adjacency_dict[candidate_node].append((new_xy, distance_val))

            return new_xy

    # אם לא נמצא אף צומת
    return None

def line_intersects_building(x1, y1, x2, y2, building_mask):
    """
    מחזיר True אם הקו בין (x1,y1) ל-(x2,y2) עובר מעל מבנה ב-building_mask.
    נשתמש באלגוריתם Bresenham על מנת לחצות את כל הפיקסלים בקו.
    building_mask[y, x] == 1 => מבנה.
    """
    points_on_line = bresenham_line_points(x1, y1, x2, y2)
    for (px, py) in points_on_line:
        # גבולות
        if py < 0 or py >= building_mask.shape[0] or px < 0 or px >= building_mask.shape[1]:
            return True  # יצא מהתמונה, נניח שזה לא תקין
        if building_mask[py, px] == 1:
            return True
    return False
