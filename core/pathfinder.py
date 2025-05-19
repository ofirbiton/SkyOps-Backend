from shared import dependencies as dep
from core.graph_builder import line_intersects_building

# Finds the shortest path between start and end using Dijkstra's algorithm
def dijkstra(start, end, adjacency_dict):
    distances = {node: float('inf') for node in adjacency_dict}
    distances[start] = 0.0

    came_from = {}
    visited = set()
    queue = [(0.0, start)]

    while queue:
        current_dist, u = dep.heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)

        if u == end:
            break

        for edge in adjacency_dict[u]:
            _, v, w = edge
            if isinstance(w, tuple):
                w = w[0]
            alt = current_dist + w
            if alt < distances[v]:
                distances[v] = alt
                came_from[v] = u
                dep.heapq.heappush(queue, (alt, v))

    if distances[end] == float('inf'):
        return None

    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = came_from[cur]
    path.reverse()
    return path

# Optimizes the path by removing unnecessary nodes (keeps only turning points and endpoints)
def optimize_path(path, building_mask):
    optimized = []
    i = 0
    n = len(path)

    while i < n:
        optimized.append(path[i])
        for j in range(n - 1, i, -1):
            x1, y1 = path[i]
            x2, y2 = path[j]
            if not line_intersects_building(x1, y1, x2, y2, building_mask):
                i = j
                break
        else:
            i += 1

    return optimized


