# improved_wumpus_world.py
import pygame
import json
import random
import sys
import heapq
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
GRID_ROWS = 5
GRID_COLS = 10
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 100
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (220, 220, 220)
DARK_GRAY = (150, 150, 150)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 200, 0)
BLUE = (0, 120, 255)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
PURPLE = (160, 32, 240)

# Load team configuration
def load_config():
    try:
        with open('team_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Error: team_config.json not found! Create it like:")
        print('{"seed": 42, "team_id": "teamX", "grid_config": {"traffic_lights":3,"cows":3,"pits":3}}')
        sys.exit(1)

# World class
class BangaloreWumpusWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        random.seed(self.seed)

        # Grid initialization
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1, 15)}
                      for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        # Agent start and state
        self.agent_start = (0, GRID_ROWS - 1)
        self.agent_pos = list(self.agent_start)
        self.agent_path = []
        self.delay_ticks = 0  # kept for future use
        self.delay_remaining = 0

        # Game state
        self.game_over = False
        self.game_won = False
        self.message = ""
        self.show_astar = True  # toggle A* visualization

        # A* visualization containers
        self.last_open_set = set()
        self.last_closed_set = set()
        self.last_path = []

        # For handling cows (avoid infinite loop): mark cells as forbidden after collision
        self.forbidden_cells = set()

        # Generate the world
        self._generate_world()

    def _generate_world(self):
        """Generate world elements with the provided seed and config."""
        num_traffic_lights = self.config['grid_config'].get('traffic_lights', 3)
        num_cows = self.config['grid_config'].get('cows', 3)
        num_pits = self.config['grid_config'].get('pits', 3)

        # Available positions excluding agent start
        available_positions = [(x, y) for x in range(GRID_COLS) for y in range(GRID_ROWS)
                               if (x, y) != tuple(self.agent_start)]
        random.shuffle(available_positions)

        # Place traffic lights
        for _ in range(num_traffic_lights):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'traffic_light'

        # Place cows
        for _ in range(num_cows):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'cow'

        # Place pits
        for _ in range(num_pits):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'pit'

        # Place goal
        if available_positions:
            goal_pos = available_positions.pop()
            self.grid[goal_pos[1]][goal_pos[0]]['type'] = 'goal'
            self.goal_pos = goal_pos
        else:
            # fallback if empty
            self.goal_pos = (GRID_COLS - 1, 0)
            self.grid[self.goal_pos[1]][self.goal_pos[0]]['type'] = 'goal'

        # Percepts generation
        self._generate_percepts()

    def _generate_percepts(self):
        """Generate percepts for each cell based on adjacent features."""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                self.grid[y][x]['percepts'] = []
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                neighbors = self._get_neighbors(x, y)
                for nx, ny in neighbors:
                    cell_type = self.grid[ny][nx]['type']
                    if cell_type == 'pit':
                        if 'breeze' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('breeze')
                    elif cell_type == 'cow':
                        if 'moo' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('moo')
                    elif cell_type == 'traffic_light':
                        if 'light' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('light')

    def _get_neighbors(self, x, y):
        """Return orthogonal neighbors only (no diagonal)."""
        neigh = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                neigh.append((nx, ny))
        return neigh

    def move_agent(self, new_x, new_y):
        """Move agent one step (non-diagonal). Handles events: traffic, cow, pit, goal."""
        if self.game_over or self.game_won:
            return

        # Validate move is orthogonal neighbor
        dx = abs(new_x - self.agent_pos[0])
        dy = abs(new_y - self.agent_pos[1])
        if dx + dy != 1:
            return

        # Check bounds
        if not (0 <= new_x < GRID_COLS and 0 <= new_y < GRID_ROWS):
            return

        # Apply move
        self.agent_pos = [new_x, new_y]
        self.agent_path.append((new_x, new_y))
        cell_type = self.grid[new_y][new_x]['type']

        # Traffic light -> set non-blocking delay
        if cell_type == 'traffic_light':
            self.message = "Waiting at traffic signal..."
            # e.g., 1.5 seconds delay
            self.delay_remaining = int(1.5 * FPS)

        # Cow -> collision: mark cell as forbidden and replan from start
        elif cell_type == 'cow':
            self.message = "Moo! Cow encountered - returning to start and forbidding that cell."
            # Mark the cow cell as forbidden so A* won't use it next time
            self.forbidden_cells.add((new_x, new_y))
            # Reset to start
            self.agent_pos = list(self.agent_start)
            self.agent_path = []
            # Game does not end - we replan when execute_path continues

        # Pit -> game over
        elif cell_type == 'pit':
            self.message = "Game Over - Fell into a pit!"
            self.game_over = True

        # Goal -> win
        elif cell_type == 'goal':
            self.message = "Goal Reached! You won!"
            self.game_won = True

    def tick(self):
        """Called every frame to handle non-blocking delays."""
        if self.delay_remaining > 0:
            self.delay_remaining -= 1
            if self.delay_remaining == 0:
                self.message = ""

    # ---------- A* Implementation ----------
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_cell_cost(self, x, y):
        """Return cost to enter cell. Pits -> inf, traffic -> high cost, cows -> high but still avoidable unless forbidden."""
        cell_type = self.grid[y][x]['type']

        if (x, y) in self.forbidden_cells:
            return float('inf')  # previously-collided cows are now forbidden

        if cell_type == 'pit':
            return float('inf')
        elif cell_type == 'traffic_light':
            return 20
        elif cell_type == 'cow':
            # allow but expensive (if not yet forbidden)
            return 50
        elif cell_type == 'goal':
            return 0
        else:
            return self.grid[y][x]['weight']

    def get_neighbors_astar(self, pos):
        """Return orthogonal neighbors for A*."""
        x, y = pos
        return self._get_neighbors(x, y)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path_astar(self):
        """
        Run A* and store visualization sets (open/closed). Returns path list or None.
        This version respects self.forbidden_cells and get_cell_cost.
        """
        start = tuple(self.agent_pos)
        goal = tuple(self.goal_pos)

        # Priority queue with (f, counter, pos)
        open_heap = []
        counter = 0
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        heapq.heappush(open_heap, (f_score[start], counter, start))
        counter += 1

        came_from = {}
        closed_set = set()
        open_set_positions = {start}

        # For visualization
        vis_open = set()
        vis_closed = set()

        while open_heap:
            current_f, _, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue

            # For visualization tracking
            vis_open = set(open_set_positions)
            vis_closed.add(current)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                self.last_open_set = vis_open
                self.last_closed_set = vis_closed
                self.last_path = path
                return path

            closed_set.add(current)
            if current in open_set_positions:
                open_set_positions.remove(current)

            for neighbor in self.get_neighbors_astar(current):
                if neighbor in closed_set:
                    continue

                move_cost = self.get_cell_cost(neighbor[0], neighbor[1])
                if move_cost == float('inf'):
                    continue  # cannot enter

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, counter, neighbor))
                    counter += 1
                    open_set_positions.add(neighbor)

        # No path found
        self.last_open_set = vis_open
        self.last_closed_set = vis_closed
        self.last_path = []
        self.message = "Path Not Found"
        return None

    def print_path_costs(self, path):
        """Print cost breakdown for debugging (Optional)."""
        if not path:
            print("No path to print.")
            return
        g = 0
        print("\nPath Cost Breakdown:")
        for pos in path:
            cell_type = self.grid[pos[1]][pos[0]]['type']
            cost = self.get_cell_cost(pos[0], pos[1])
            h = self.heuristic(pos, self.goal_pos)
            print(f"{pos} -> type={cell_type} cost_enter={cost} g_so_far={g} h={h}")
            if cost != float('inf'):
                g += cost

    def execute_path(self, path):
        """
        Execute the path step by step. If a cow collision occurs (forbidden added),
        we re-run A* until success or no path exists.
        """
        if path is None:
            return

        # Make a deque of steps (skip if already at start)
        steps = deque(path)
        # If agent isn't at the path's first node, ensure we start from path[0] being agent pos
        if steps and tuple(self.agent_pos) != steps[0]:
            # If agent is at start, ensure the path begins at start. If not, prepend current pos.
            if tuple(self.agent_pos) not in steps:
                steps.appendleft(tuple(self.agent_pos))

        while steps and not (self.game_over or self.game_won):
            next_pos = steps.popleft()
            # skip staying in same cell if path included start
            if tuple(self.agent_pos) == next_pos:
                continue

            # Wait handled per frame by tick(); here call move
            self.move_agent(next_pos[0], next_pos[1])
            # If a cow collision occurred we reset and need to replan
            if tuple(self.agent_pos) == self.agent_start and self.forbidden_cells:
                # Try to replan from start
                new_path = self.find_path_astar()
                if new_path is None:
                    # no path exists
                    return
                # update steps to new path (skip the first node as agent is at start)
                steps = deque(new_path)
                if steps and steps[0] == tuple(self.agent_pos):
                    steps.popleft()
            # If traffic light delay is set, wait frames until cleared (non-blocking)
            while self.delay_remaining > 0 and not (self.game_over or self.game_won):
                # The game loop will call tick() and render, but here we'll break
                # Execution will return to main loop so we shouldn't busy-wait here.
                return  # return to main loop to allow UI ticks

    # ----------------- NEW: get_current_percepts -----------------
    def get_current_percepts(self):
        """
        Return the list of percepts at the agent's current location.
        Uses the percepts already generated in the grid cells.
        """
        x, y = self.agent_pos
        # Safety: ensure coords are valid
        if not (0 <= x < GRID_COLS and 0 <= y < GRID_ROWS):
            return []
        return list(self.grid[y][x].get('percepts', []))

# Renderer class
class GameRenderer:
    def __init__(self, world):
        self.world = world
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bangalore Wumpus World - Improved")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, WINDOW_HEIGHT - 100), 2)
        for y in range(0, WINDOW_HEIGHT - 100, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)

    def draw_cell_contents(self):
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                cell = self.world.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE

                # base rect
                pygame.draw.rect(self.screen, WHITE, (px+1, py+1, CELL_SIZE-2, CELL_SIZE-2))

                # draw types
                if cell['type'] == 'traffic_light':
                    pygame.draw.circle(self.screen, RED, (px + CELL_SIZE//2, py + CELL_SIZE//2), 18)
                    txt = self.small_font.render("SIGNAL", True, WHITE)
                    self.screen.blit(txt, (px + 8, py + CELL_SIZE//2 + 22))
                elif cell['type'] == 'cow':
                    pygame.draw.rect(self.screen, BROWN, (px + 18, py + 18, 44, 44))
                    txt = self.small_font.render("COW", True, WHITE)
                    self.screen.blit(txt, (px + 25, py + 32))
                elif cell['type'] == 'pit':
                    pygame.draw.circle(self.screen, BLACK, (px + CELL_SIZE//2, py + CELL_SIZE//2), 24)
                    txt = self.small_font.render("PIT", True, WHITE)
                    self.screen.blit(txt, (px + 30, py + 30))
                elif cell['type'] == 'goal':
                    pygame.draw.rect(self.screen, GREEN, (px + 14, py + 14, 52, 52))
                    txt = self.small_font.render("GOAL", True, BLACK)
                    self.screen.blit(txt, (px + 20, py + 30))

                # Forbidden (after cow collision) highlight
                if (x, y) in self.world.forbidden_cells:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    s.fill((255, 0, 0, 80))
                    self.screen.blit(s, (px, py))

                # percept indicators (small)
                pyoff = 6
                if 'breeze' in cell['percepts']:
                    text = self.small_font.render("~", True, BLUE)
                    self.screen.blit(text, (px + 4, py + pyoff)); pyoff += 14
                if 'moo' in cell['percepts']:
                    text = self.small_font.render("M", True, BROWN)
                    self.screen.blit(text, (px + 4, py + pyoff)); pyoff += 14
                if 'light' in cell['percepts']:
                    text = self.small_font.render("L", True, ORANGE)
                    self.screen.blit(text, (px + 4, py + pyoff))

                # show weight for normal cells for debugging
                if cell['type'] == 'empty':
                    wt_txt = self.small_font.render(str(cell['weight']), True, DARK_GRAY)
                    self.screen.blit(wt_txt, (px + CELL_SIZE - 24, py + CELL_SIZE - 22))

    def draw_astar_visualization(self):
        """Draw open set (blue), closed set (gray) and path (yellow)."""
        # closed
        for pos in self.world.last_closed_set:
            x, y = pos
            px = x * CELL_SIZE
            py = y * CELL_SIZE
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((100, 100, 100, 80))
            self.screen.blit(s, (px, py))
        # open
        for pos in self.world.last_open_set:
            x, y = pos
            px = x * CELL_SIZE
            py = y * CELL_SIZE
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((0, 100, 255, 60))
            self.screen.blit(s, (px, py))
        # path
        for pos in self.world.last_path:
            x, y = pos
            px = x * CELL_SIZE
            py = y * CELL_SIZE
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((255, 240, 80, 110))
            self.screen.blit(s, (px, py))

    def draw_agent(self):
        x, y = self.world.agent_pos
        px = x * CELL_SIZE + CELL_SIZE // 2
        py = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, YELLOW, (px, py), 15)
        pygame.draw.circle(self.screen, BLACK, (px, py), 15, 2)
        pygame.draw.circle(self.screen, BLACK, (px - 5, py - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (px + 5, py - 3), 3)

    def draw_info(self):
        info_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, GRAY, (0, info_y, WINDOW_WIDTH, 100))
        pos_text = self.font.render(f"Position: {self.world.agent_pos}", True, BLACK)
        self.screen.blit(pos_text, (10, info_y + 8))
        percepts = self.world.get_current_percepts()
        percept_text = self.font.render(f"Percepts: {', '.join(percepts) if percepts else 'None'}", True, BLACK)
        self.screen.blit(percept_text, (10, info_y + 32))
        msg_text = self.font.render(self.world.message, True, RED if self.world.game_over else GREEN)
        self.screen.blit(msg_text, (10, info_y + 56))
        # Controls hint
        hint = self.small_font.render("SPACE: A*  |  V: Toggle A* viz  |  R: Reset  |  C: Clear forbidden cells", True, BLACK)
        self.screen.blit(hint, (350, info_y + 40))

    def render(self):
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_cell_contents()
        if self.world.show_astar:
            self.draw_astar_visualization()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(FPS)

# Main game loop
def main():
    config = load_config()
    world = BangaloreWumpusWorld(config)
    renderer = GameRenderer(world)

    print("=== Bangalore Wumpus World (Improved) ===")
    print(f"Team ID: {config.get('team_id')}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    print("Controls: Arrow keys manual | SPACE run A* | V toggle viz | R reset | C clear forbidden | ESC quit")

    running = True
    pending_execution = None  # holds the current path to execute

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    world = BangaloreWumpusWorld(config)
                    renderer.world = world
                    pending_execution = None

                elif event.key == pygame.K_c:
                    # Clear forbidden cells (allow trying again)
                    world.forbidden_cells.clear()
                    world.message = "Cleared forbidden cells."
                    world.last_open_set = set()
                    world.last_closed_set = set()
                    world.last_path = []

                elif event.key == pygame.K_v:
                    world.show_astar = not world.show_astar

                elif event.key == pygame.K_SPACE:
                    # Execute A* and set pending path for stepwise execution
                    print("\n=== Executing A* Pathfinding ===")
                    path = world.find_path_astar()
                    if path:
                        print(f"Path found: {path}")
                        world.print_path_costs(path)
                        pending_execution = deque(path)
                    else:
                        print("Path not found!")
                        pending_execution = None

                elif event.key == pygame.K_UP:
                    world.move_agent(world.agent_pos[0], world.agent_pos[1] - 1)
                elif event.key == pygame.K_DOWN:
                    world.move_agent(world.agent_pos[0], world.agent_pos[1] + 1)
                elif event.key == pygame.K_LEFT:
                    world.move_agent(world.agent_pos[0] - 1, world.agent_pos[1])
                elif event.key == pygame.K_RIGHT:
                    world.move_agent(world.agent_pos[0] + 1, world.agent_pos[1])

        # Tick (handle non-blocking timers)
        world.tick()

        # If there's a pending path to follow, advance one step per frame (keeps UI responsive)
        if pending_execution and not (world.game_over or world.game_won):
            # If agent not at path's first node, ensure it starts properly
            if pending_execution:
                next_pos = pending_execution[0]
                # If the pending path's first element equals agent current, pop it
                if tuple(world.agent_pos) == next_pos:
                    pending_execution.popleft()
                if pending_execution:
                    # If currently waiting at traffic light, don't move
                    if world.delay_remaining > 0:
                        # wait until delay clears
                        pass
                    else:
                        # move to the next cell
                        np = pending_execution.popleft()
                        world.move_agent(np[0], np[1])
                        # If cow forced forbidden and reset happened, we must replan
                        if tuple(world.agent_pos) == world.agent_start and world.forbidden_cells:
                            new_path = world.find_path_astar()
                            if new_path:
                                pending_execution = deque(new_path)
                            else:
                                pending_execution = None

        renderer.render()

    pygame.quit()

if __name__ == "__main__":
    main()
