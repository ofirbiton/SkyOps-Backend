from shared import dependencies as dep

def dijkstra(start, end, adjacency_dict):
    """
    Runs Dijkstra's algorithm and returns a path [(x, y), (x, y), ...] from start to end.
    If no path is found, returns None.
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
            # Each edge is expected to be in the form: [ (x1, y1), (x2, y2), dist ]
            _, v, w = edge
            if isinstance(w, tuple):  # If weight is a tuple, extract the first element
                w = w[0]
            alt = current_dist + w
            if alt < distances[v]:
                distances[v] = alt
                came_from[v] = u
                dep.heapq.heappush(queue, (alt, v))

    if distances[end] == float('inf'):
        return None  # No route found

    # Reconstruct path
    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = came_from[cur]
    path.reverse()
    return path

