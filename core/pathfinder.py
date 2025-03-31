from shared import dependencies as dep

def dijkstra(start, end, adjacency_dict):
    """
    מפעיל את אלגוריתם דייקסטרה ומחזיר מסלול [(x, y), (x, y), ...] מהצומת start עד end.
    אם אין מסלול, מחזיר None.
    """
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
            alt = current_dist + w
            if alt < distances[v]:
                distances[v] = alt
                came_from[v] = u
                dep.heapq.heappush(queue, (alt, v))

    if distances[end] == float('inf'):
        return None  # אין מסלול

    # שחזור המסלול
    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = came_from[cur]
    path.reverse()
    return path
