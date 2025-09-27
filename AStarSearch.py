import heapq

class AStarSearch:
  def __init__(self, position, g, h):  
    self.position = position
    self.g = g
    self.h = h
    self.f = g + h
    self.parent = None                

  def __lt__(self, other):
    return self.f < other.f

def AStar(grid, start, goal):
  open_list = []
  closed_list = set()

  start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
  start_node = AStarSearch(start, 0, start_h)  
  heapq.heappush(open_list, start_node)

  while open_list:
    current_node = heapq.heappop(open_list)

    if current_node.position == goal:
      path = []
      while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
      return path[::-1]

    closed_list.add(current_node.position)

    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
      nr, nc = current_node.position[0] + dr, current_node.position[1] + dc
      neigh_pos = (nr, nc)

      in_bound = 0 <= nr < len(grid) and 0 <= nc < len(grid[0])
      if not in_bound: continue
      if grid[nr][nc] == "X": continue
      if neigh_pos in closed_list: continue

      neigh_g = current_node.g + 1
      neigh_h = abs(neigh_pos[0] - goal[0]) + abs(neigh_pos[1] - goal[1])
      neigh_node = AStarSearch(neigh_pos, neigh_g, neigh_h)  
      neigh_node.parent = current_node
      heapq.heappush(open_list, neigh_node)

  return None  
