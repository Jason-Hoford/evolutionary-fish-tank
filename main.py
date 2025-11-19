import math
import random
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pygame


# --- Simulation constants ---

WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000

VIEW_WIDTH = 1000
VIEW_HEIGHT = 600

INITIAL_FISH_COUNT = 40
MAX_FOOD = 120

FPS = 60

# Fixed Y-scale for the food consumption graph (food per simulated second)
MAX_CONSUMPTION_RATE = 10.0


# --- Utility functions ---


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def wrap_position(x: float, y: float) -> Tuple[float, float]:
    """Deprecated: world no longer wraps; kept for backwards compatibility."""
    return clamp(x, 0, WORLD_WIDTH), clamp(y, 0, WORLD_HEIGHT)


def nn_to_color(weights: List[np.ndarray]) -> Tuple[int, int, int]:
    """
    Derive a stable color from a list of weight matrices.
    Color similarity corresponds to network similarity.
    """
    flat = np.concatenate([w.flatten() for w in weights])
    if flat.size == 0:
        return 128, 128, 128

    # More sensitive projections to 3 channels:
    # combine multiple weighted sums and use sharper nonlinearities
    # so that small NN differences show up clearly in color space.
    s0 = float(flat[::3].sum())
    s1 = float(flat[1::3].sum())
    s2 = float(flat[2::3].sum())
    s3 = float(flat[::5].sum())
    s4 = float(flat[2::7].sum())

    r = math.tanh(0.06 * s0 + 0.04 * s3)
    g = math.tanh(0.06 * s1 - 0.04 * s4)
    b = math.tanh(0.06 * s2 + 0.02 * s0 - 0.02 * s1)

    r = int((r * 0.5 + 0.5) * 255)
    g = int((g * 0.5 + 0.5) * 255)
    b = int((b * 0.5 + 0.5) * 255)

    return int(clamp(r, 0, 255)), int(clamp(g, 0, 255)), int(clamp(b, 0, 255))


def format_sim_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# --- Neural network ---


class FeedForwardNet:
    """
    Small fully-connected feedforward network.
    Used as the fish's policy: observation -> movement decisions.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            # Xavier initialization
            limit = math.sqrt(6 / (in_size + out_size))
            w = np.random.uniform(-limit, limit, (out_size, in_size)).astype(np.float32)
            b = np.zeros(out_size, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def clone(self) -> "FeedForwardNet":
        clone = FeedForwardNet(self.layer_sizes[0], self.layer_sizes[1:-1], self.layer_sizes[-1])
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        return clone

    def mutate(self, rate: float = 0.05, scale: float = 0.2) -> None:
        """
        Apply small random Gaussian mutations to weights and biases.
        This is our simple evolutionary RL mechanism.
        """
        for i in range(len(self.weights)):
            mask_w = np.random.rand(*self.weights[i].shape) < rate
            noise_w = np.random.normal(0.0, scale, self.weights[i].shape).astype(np.float32)
            self.weights[i] += mask_w * noise_w

            mask_b = np.random.rand(*self.biases[i].shape) < rate
            noise_b = np.random.normal(0.0, scale, self.biases[i].shape).astype(np.float32)
            self.biases[i] += mask_b * noise_b

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (input_size,)
        returns: shape (output_size,)
        """
        a = x
        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]
            if i < len(self.weights) - 1:
                # Hidden layers: ReLU
                a = np.maximum(0, z)
            else:
                # Output layer: tanh for smooth actions
                a = np.tanh(z)
        return a


# --- Entities ---


@dataclass
class Food:
    x: float
    y: float
    radius: float = 4.0


@dataclass
class Fish:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    speed: float
    net: FeedForwardNet
    hunger: float = 0.0
    age: float = 0.0
    energy: float = 0.0  # used to determine reproduction
    alive: bool = True
    size: float = 6.0  # visual radius, grows with age
    mature_age: float = 25.0
    max_size: float = 16.0

    def update(self, dt: float, nearest_food: Food | None) -> None:
        if not self.alive:
            return

        self.age += dt

        # Growth until maturity
        if self.age < self.mature_age:
            growth_factor = self.age / self.mature_age
            self.size = 6.0 + (self.max_size - 6.0) * growth_factor
        else:
            self.size = self.max_size

        # Observation: relative vector to nearest food, velocity, hunger, age
        # All normalized to roughly [-1, 1]
        if nearest_food is not None:
            dx = nearest_food.x - self.x
            dy = nearest_food.y - self.y
        else:
            dx, dy = 0.0, 0.0

        dist = math.sqrt(dx * dx + dy * dy) + 1e-5
        dir_to_food_x = dx / dist
        dir_to_food_y = dy / dist

        speed_norm = self.speed / 200.0
        vx_norm = self.vx / 200.0
        vy_norm = self.vy / 200.0

        hunger_norm = clamp(self.hunger / 100.0, 0.0, 1.0) * 2 - 1
        age_norm = clamp(self.age / 100.0, 0.0, 1.0) * 2 - 1
        size_norm = clamp((self.size - 6.0) / (self.max_size - 6.0), 0.0, 1.0) * 2 - 1

        obs = np.array(
            [
                dir_to_food_x,
                dir_to_food_y,
                vx_norm,
                vy_norm,
                hunger_norm,
                age_norm,
                size_norm,
            ],
            dtype=np.float32,
        )

        action = self.net.forward(obs)
        # action[0]: turn (-1,1), action[1]: throttle (-1,1)
        turn = float(action[0])
        throttle = float(action[1])

        # Update angle and speed
        self.angle += turn * 3.0 * dt  # radians per second
        self.speed = clamp(self.speed + throttle * 80.0 * dt, 10.0, 200.0)

        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off real tank walls instead of wrapping
        if self.x < 0:
            self.x = 0
            self.vx *= -1
            self.angle = math.pi - self.angle
        elif self.x > WORLD_WIDTH:
            self.x = WORLD_WIDTH
            self.vx *= -1
            self.angle = math.pi - self.angle

        if self.y < 0:
            self.y = 0
            self.vy *= -1
            self.angle = -self.angle
        elif self.y > WORLD_HEIGHT:
            self.y = WORLD_HEIGHT
            self.vy *= -1
            self.angle = -self.angle

        # Hunger increases; larger fish pay higher cost
        hunger_rate = 2.0 + (self.size / self.max_size) * 3.0
        self.hunger += hunger_rate * dt

        if self.hunger > 100.0:
            self.alive = False

    def try_eat(self, foods: List[Food]) -> int:
        """Return number of pellets eaten."""
        if not self.alive:
            return 0

        eaten = 0
        eat_radius = self.size + 4.0
        remaining_foods = []
        for food in foods:
            dx = food.x - self.x
            dy = food.y - self.y

            if dx * dx + dy * dy <= eat_radius * eat_radius:
                eaten += 1
            else:
                remaining_foods.append(food)

        if eaten > 0:
            # Eating reduces hunger and increases energy (for reproduction)
            self.hunger = max(0.0, self.hunger - 15.0 * eaten)
            self.energy += 20.0 * eaten

        foods[:] = remaining_foods
        return eaten

    def can_reproduce(self) -> bool:
        return self.alive and self.energy >= 60.0 and self.age > self.mature_age * 0.7


# --- World / Simulation ---


class World:
    def __init__(self):
        self.fish: List[Fish] = []
        self.food: List[Food] = []
        # Timestamps (simulation time) when eaten food should respawn
        self.pending_food_respawns: list[float] = []

        self._init_population()

    def _init_population(self):
        for _ in range(INITIAL_FISH_COUNT):
            net = FeedForwardNet(input_size=7, hidden_sizes=[16, 16], output_size=2)
            x = random.uniform(0, WORLD_WIDTH)
            y = random.uniform(0, WORLD_HEIGHT)
            angle = random.uniform(0, math.tau)
            speed = random.uniform(40, 120)
            fish = Fish(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                angle=angle,
                speed=speed,
                net=net,
            )
            self.fish.append(fish)

        for _ in range(MAX_FOOD):
            self.spawn_food()

    def spawn_food(self):
        x = random.uniform(0, WORLD_WIDTH)
        y = random.uniform(0, WORLD_HEIGHT)
        self.food.append(Food(x=x, y=y))

    def find_nearest_food(self, fish: Fish) -> Food | None:
        nearest = None
        best_dist_sq = float("inf")
        for food in self.food:
            dx = food.x - fish.x
            dy = food.y - fish.y

            d_sq = dx * dx + dy * dy
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                nearest = food
        return nearest

    def update(self, dt: float, sim_time: float) -> int:
        total_eaten = 0
        # Update fish behavior
        for f in self.fish:
            nearest_food = self.find_nearest_food(f)
            f.update(dt, nearest_food)

        # Simple fish-fish collision handling: push overlapping fish apart
        n = len(self.fish)
        for i in range(n):
            fi = self.fish[i]
            if not fi.alive:
                continue
            for j in range(i + 1, n):
                fj = self.fish[j]
                if not fj.alive:
                    continue

                dx = fj.x - fi.x
                dy = fj.y - fi.y

                min_dist = fi.size + fj.size
                dist_sq = dx * dx + dy * dy
                if dist_sq > 0.0 and dist_sq < min_dist * min_dist:
                    dist = math.sqrt(dist_sq)
                    # Normalized separation direction
                    nx = dx / dist
                    ny = dy / dist
                    overlap = min_dist - dist
                    # Push each fish half the overlap distance away from the other
                    fi.x -= nx * overlap * 0.5
                    fi.y -= ny * overlap * 0.5
                    fj.x += nx * overlap * 0.5
                    fj.y += ny * overlap * 0.5
                    fi.x = clamp(fi.x, 0, WORLD_WIDTH)
                    fi.y = clamp(fi.y, 0, WORLD_HEIGHT)
                    fj.x = clamp(fj.x, 0, WORLD_WIDTH)
                    fj.y = clamp(fj.y, 0, WORLD_HEIGHT)

        # Eating
        for f in self.fish:
            eaten = f.try_eat(self.food)
            total_eaten += eaten
            # Schedule respawn for each eaten pellet 10 seconds later
            if eaten > 0:
                for _ in range(eaten):
                    self.pending_food_respawns.append(sim_time + 10.0)

        # Reproduction
        new_fish: List[Fish] = []
        for f in self.fish:
            if f.can_reproduce():
                f.energy *= 0.4  # reproduction cost
                child_net = f.net.clone()
                # Mutate child network
                child_net.mutate(rate=0.05, scale=0.15)
                # Small position offset
                angle = random.uniform(0, math.tau)
                dist = random.uniform(5, 20)
                child_x = clamp(f.x + math.cos(angle) * dist, 0, WORLD_WIDTH)
                child_y = clamp(f.y + math.sin(angle) * dist, 0, WORLD_HEIGHT)
                child_angle = random.uniform(0, math.tau)
                child_speed = random.uniform(40, 120)
                child = Fish(
                    x=child_x,
                    y=child_y,
                    vx=math.cos(child_angle) * child_speed,
                    vy=math.sin(child_angle) * child_speed,
                    angle=child_angle,
                    speed=child_speed,
                    net=child_net,
                    hunger=0.0,
                    age=0.0,
                    energy=10.0,
                    size=6.0,
                )
                new_fish.append(child)

        self.fish.extend(new_fish)

        # Cull dead fish; if everyone dies, respawn a new population
        self.fish = [f for f in self.fish if f.alive]
        if not self.fish:
            self._init_population()

        # Respawn any food whose delay has elapsed
        if self.pending_food_respawns:
            remaining: list[float] = []
            for t in self.pending_food_respawns:
                if t <= sim_time and len(self.food) < MAX_FOOD:
                    self.spawn_food()
                else:
                    remaining.append(t)
            self.pending_food_respawns = remaining

        return total_eaten


# --- Rendering / Save/Load / Main loop ---


def draw_world(
    screen: pygame.Surface,
    world: World,
    camera_pos: Tuple[float, float],
    zoom: float,
    show_graph: bool,
    consumption_history: list[tuple[float, float]],
    sim_time: float,
    time_scale: float,
):
    cx, cy = camera_pos

    screen.fill((5, 5, 20))

    def world_to_screen(wx: float, wy: float) -> tuple[float, float]:
        # Convert world coordinates to screen coordinates with zoom and camera
        sx = (wx - cx) * zoom + VIEW_WIDTH / 2
        sy = (wy - cy) * zoom + VIEW_HEIGHT / 2
        return sx, sy

    # Draw food
    for food in world.food:
        fx, fy = world_to_screen(food.x, food.y)

        if -10 <= fx <= VIEW_WIDTH + 10 and -10 <= fy <= VIEW_HEIGHT + 10:
            pygame.draw.circle(
                screen, (200, 200, 80), (int(fx), int(fy)), max(1, int(food.radius * zoom))
            )

    # Draw fish
    for fish in world.fish:
        fx, fy = world_to_screen(fish.x, fish.y)

        if -30 <= fx <= VIEW_WIDTH + 30 and -30 <= fy <= VIEW_HEIGHT + 30:
            color = nn_to_color(fish.net.weights)
            body_radius = max(2, int(fish.size * zoom))

            # Triangle pointing in movement direction
            angle = fish.angle
            nose = (fx + math.cos(angle) * body_radius, fy + math.sin(angle) * body_radius)
            left = (
                fx + math.cos(angle + 2.5) * body_radius * 0.7,
                fy + math.sin(angle + 2.5) * body_radius * 0.7,
            )
            right = (
                fx + math.cos(angle - 2.5) * body_radius * 0.7,
                fy + math.sin(angle - 2.5) * body_radius * 0.7,
            )
            pygame.draw.polygon(screen, color, [nose, left, right])

    font = pygame.font.Font(None, 24)

    # Optional consumption graph for last 10 seconds
    if show_graph and consumption_history:
        graph_width = 260
        graph_height = 100
        margin = 10
        x0 = VIEW_WIDTH - graph_width - margin
        y0 = VIEW_HEIGHT - graph_height - margin

        pygame.draw.rect(screen, (0, 0, 0), (x0, y0, graph_width, graph_height))
        pygame.draw.rect(screen, (80, 80, 80), (x0, y0, graph_width, graph_height), 1)

        recent = [(t, r) for (t, r) in consumption_history if sim_time - t <= 10.0]
        if len(recent) >= 2:
            max_rate = MAX_CONSUMPTION_RATE
            min_time = max(sim_time - 10.0, recent[0][0])
            time_span = 10.0

            # Axes: x = time (last 10s), y = food/sec
            axis_color = (120, 120, 120)
            # Y-axis
            pygame.draw.line(
                screen,
                axis_color,
                (x0 + 30, y0 + 10),
                (x0 + 30, y0 + graph_height - 20),
                1,
            )
            # X-axis
            pygame.draw.line(
                screen,
                axis_color,
                (x0 + 30, y0 + graph_height - 20),
                (x0 + graph_width - 10, y0 + graph_height - 20),
                1,
            )

            # Grid lines
            grid_color = (40, 40, 70)
            # Vertical grid every 2 seconds
            for step in range(0, 11, 2):
                frac = step / time_span
                gx = x0 + 30 + frac * (graph_width - 40)
                pygame.draw.line(
                    screen,
                    grid_color,
                    (gx, y0 + 10),
                    (gx, y0 + graph_height - 20),
                    1,
                )
            # Horizontal grid every 2 food/sec
            for rate in range(0, int(max_rate) + 1, 2):
                frac = rate / max_rate if max_rate else 0.0
                gy = y0 + graph_height - 20 - frac * (graph_height - 40)
                pygame.draw.line(
                    screen,
                    grid_color,
                    (x0 + 30, gy),
                    (x0 + graph_width - 10, gy),
                    1,
                )

            # Smoother line via exponential smoothing
            smoothed = []
            smooth_val = min(recent[0][1], max_rate)
            alpha = 0.3
            for _, rate in recent:
                rate = min(rate, max_rate)
                smooth_val = (1 - alpha) * smooth_val + alpha * rate
                smoothed.append(smooth_val)

            points = []
            for (t, _), smooth_rate in zip(recent, smoothed):
                x_norm = (t - min_time) / time_span
                y_norm = smooth_rate / max_rate if max_rate else 0.0
                px = x0 + 30 + x_norm * (graph_width - 40)
                py = y0 + graph_height - 20 - y_norm * (graph_height - 40)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.aalines(screen, (80, 200, 255), False, points, blend=1)

            # Axis labels and ticks
            rate_label = font.render("food/sec", True, (200, 200, 200))
            screen.blit(rate_label, (x0 + 5, y0 + 5))

            # Y ticks: 0 and fixed max_rate
            for frac, label_val in ((0.0, 0.0), (1.0, max_rate)):
                y_tick = y0 + graph_height - 20 - frac * (graph_height - 40)
                pygame.draw.line(
                    screen,
                    axis_color,
                    (x0 + 28, int(y_tick)),
                    (x0 + 32, int(y_tick)),
                    1,
                )
                label = font.render(f"{label_val:.1f}", True, (200, 200, 200))
                screen.blit(label, (x0 + 35, int(y_tick) - 8))

            # X ticks: -10s and 0s (now)
            for frac, label_txt in ((0.0, "-10s"), (1.0, "0s")):
                x_tick = x0 + 30 + frac * (graph_width - 40)
                pygame.draw.line(
                    screen,
                    axis_color,
                    (int(x_tick), y0 + graph_height - 22),
                    (int(x_tick), y0 + graph_height - 18),
                    1,
                )
                label = font.render(label_txt, True, (200, 200, 200))
                screen.blit(label, (int(x_tick) - 15, y0 + graph_height - 18))

    draw_hud(screen, world, zoom, time_scale, sim_time, show_graph)

    # Draw visual border corresponding to the real tank edges (world border)
    border_color = (90, 90, 160)
    # World corners in screen space
    top_left = world_to_screen(0, 0)
    bottom_right = world_to_screen(WORLD_WIDTH, WORLD_HEIGHT)
    rect_x = min(top_left[0], bottom_right[0])
    rect_y = min(top_left[1], bottom_right[1])
    rect_w = abs(bottom_right[0] - top_left[0])
    rect_h = abs(bottom_right[1] - top_left[1])
    border_rect = pygame.Rect(rect_x, rect_y, rect_w, rect_h)
    pygame.draw.rect(screen, border_color, border_rect, 3)


def draw_hud(
    screen: pygame.Surface,
    world: World,
    zoom: float,
    time_scale: float,
    sim_time: float,
    show_graph: bool,
):
    hud_surface = pygame.Surface((280, 150), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 160))
    font = pygame.font.Font(None, 24)

    lines = [
        "NEURAL FISH TANK",
        f"Fish: {len(world.fish)}",
        f"Food: {len(world.food)} (pending {len(world.pending_food_respawns)})",
        f"Zoom: {zoom:.2f}x",
        f"Speed: {time_scale:.1f}x",
        f"Time: {format_sim_time(sim_time)}",
        f"Graph: {'ON' if show_graph else 'OFF'}",
    ]

    y = 10
    for i, line in enumerate(lines):
        color = (255, 210, 120) if i == 0 else (220, 220, 230)
        text = font.render(line, True, color)
        hud_surface.blit(text, (12, y))
        y += 20

    hud_width = hud_surface.get_width()
    screen.blit(hud_surface, (VIEW_WIDTH - hud_width - 10, 10))

    # Instruction ribbon at the bottom
    ribbon_surface = pygame.Surface((VIEW_WIDTH - 20, 34), pygame.SRCALPHA)
    ribbon_surface.fill((0, 0, 0, 150))
    hint_font = pygame.font.Font(None, 22)
    hint_text = (
        "WASD move  |  QE zoom  |  1-4 speed  |  SPACE focus  |  G graph  |  F5 save  |  F9 load  |  ESC menu"
    )
    text_render = hint_font.render(hint_text, True, (200, 200, 210))
    ribbon_surface.blit(text_render, (10, 8))
    screen.blit(ribbon_surface, (10, VIEW_HEIGHT - 44))


def draw_menu(
    screen: pygame.Surface,
    options: list[str],
    selected_index: int,
    message: str,
    has_running_sim: bool,
):
    screen.fill((8, 10, 22))

    title_font = pygame.font.Font(None, 64)
    option_font = pygame.font.Font(None, 36)
    message_font = pygame.font.Font(None, 28)

    title = title_font.render("Neural Fish Tank", True, (240, 240, 255))
    subtitle = message_font.render(
        "Select an option to begin", True, (190, 190, 210)
    )

    screen.blit(title, (VIEW_WIDTH / 2 - title.get_width() / 2, 100))
    screen.blit(subtitle, (VIEW_WIDTH / 2 - subtitle.get_width() / 2, 160))

    start_y = 230
    for idx, option in enumerate(options):
        is_selected = idx == selected_index
        bg_rect = pygame.Rect(
            VIEW_WIDTH / 2 - 200,
            start_y + idx * 60,
            400,
            50,
        )
        color = (40, 60, 110) if is_selected else (20, 30, 60)
        pygame.draw.rect(screen, color, bg_rect, border_radius=8)
        pygame.draw.rect(screen, (120, 150, 200), bg_rect, 2, border_radius=8)

        label = option_font.render(option, True, (240, 240, 255))
        screen.blit(
            label,
            (
                VIEW_WIDTH / 2 - label.get_width() / 2,
                start_y + idx * 60 + 10,
            ),
        )

    if message:
        msg = message_font.render(message, True, (255, 180, 140))
        screen.blit(msg, (VIEW_WIDTH / 2 - msg.get_width() / 2, start_y + len(options) * 60 + 20))

    if has_running_sim:
        hint = message_font.render("Press ESC during a run to return to this menu.", True, (180, 180, 200))
        screen.blit(hint, (VIEW_WIDTH / 2 - hint.get_width() / 2, VIEW_HEIGHT - 80))

    footer = message_font.render("Use ↑/↓ and Enter to choose. Esc quits.", True, (160, 160, 180))
    screen.blit(footer, (VIEW_WIDTH / 2 - footer.get_width() / 2, VIEW_HEIGHT - 40))


def save_state(
    path: str,
    world: World,
    camera_x: float,
    camera_y: float,
    zoom: float,
    time_scale: float,
    sim_time: float,
    consumption_history: list[tuple[float, float]],
):
    state = {
        "world": world,
        "camera_x": camera_x,
        "camera_y": camera_y,
        "zoom": zoom,
        "time_scale": time_scale,
        "sim_time": sim_time,
        "consumption_history": consumption_history,
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)
    return (
        state["world"],
        state["camera_x"],
        state["camera_y"],
        state["zoom"],
        state["time_scale"],
        state["sim_time"],
        state.get("consumption_history", []),
    )


def main():
    pygame.init()
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption("Neural Fish Tank - RL & Evolution")

    clock = pygame.time.Clock()
    world: World | None = None

    camera_x = WORLD_WIDTH / 2
    camera_y = WORLD_HEIGHT / 2
    zoom = 1.0
    time_scale = 1.0
    sim_time = 0.0
    show_graph = False
    consumption_history: list[tuple[float, float]] = []

    mode = "menu"  # "menu" or "running"
    menu_index = 0
    menu_message = ""

    def reset_world():
        nonlocal world, camera_x, camera_y, zoom, time_scale, sim_time, show_graph, consumption_history
        world = World()
        camera_x = WORLD_WIDTH / 2
        camera_y = WORLD_HEIGHT / 2
        zoom = 1.0
        time_scale = 1.0
        sim_time = 0.0
        show_graph = False
        consumption_history = []

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        if mode == "menu":
            menu_options = []
            if world is not None:
                menu_options.append("Resume Simulation")
            menu_options.extend(["Start New Simulation", "Load Saved Simulation", "Quit"])
            if menu_index >= len(menu_options):
                menu_index = max(0, len(menu_options) - 1)
        else:
            menu_options = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif mode == "menu" and event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    menu_index = (menu_index - 1) % len(menu_options)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    menu_index = (menu_index + 1) % len(menu_options)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    choice = menu_options[menu_index]
                    if choice == "Resume Simulation" and world is not None:
                        mode = "running"
                        menu_message = ""
                    elif choice == "Start New Simulation":
                        reset_world()
                        mode = "running"
                        menu_message = ""
                    elif choice == "Load Saved Simulation":
                        try:
                            (
                                world,
                                camera_x,
                                camera_y,
                                zoom,
                                time_scale,
                                sim_time,
                                consumption_history,
                            ) = load_state("fish_tank_save.pkl")
                            show_graph = False
                            mode = "running"
                            menu_message = ""
                            print("Loaded state from fish_tank_save.pkl")
                        except Exception as e:
                            menu_message = f"Failed to load save: {e}"
                            print(menu_message)
                    elif choice == "Quit":
                        running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif mode == "running" and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    show_graph = not show_graph
                elif event.key == pygame.K_F5 and world is not None:
                    try:
                        save_state(
                            "fish_tank_save.pkl",
                            world,
                            camera_x,
                            camera_y,
                            zoom,
                            time_scale,
                            sim_time,
                            consumption_history,
                        )
                        print("Saved state to fish_tank_save.pkl")
                    except Exception as e:
                        print(f"Failed to save state: {e}")
                elif event.key == pygame.K_F9:
                    try:
                        (
                            world,
                            camera_x,
                            camera_y,
                            zoom,
                            time_scale,
                            sim_time,
                            consumption_history,
                        ) = load_state("fish_tank_save.pkl")
                        show_graph = False
                        print("Loaded state from fish_tank_save.pkl")
                    except Exception as e:
                        print(f"Failed to load state: {e}")
                elif event.key == pygame.K_ESCAPE:
                    mode = "menu"
                    menu_message = "Simulation paused."

        if mode == "running" and world is not None:
            keys = pygame.key.get_pressed()
            cam_speed = 400.0
            if keys[pygame.K_a]:
                camera_x -= cam_speed * dt
            if keys[pygame.K_d]:
                camera_x += cam_speed * dt
            if keys[pygame.K_w]:
                camera_y -= cam_speed * dt
            if keys[pygame.K_s]:
                camera_y += cam_speed * dt

            # Zoom controls
            if keys[pygame.K_q]:
                zoom /= 1.02
            if keys[pygame.K_e]:
                zoom *= 1.02
            zoom = clamp(zoom, 0.3, 3.0)

            # Time scaling controls
            if keys[pygame.K_1]:
                time_scale = 0.5
            if keys[pygame.K_2]:
                time_scale = 1.0
            if keys[pygame.K_3]:
                time_scale = 2.0
            if keys[pygame.K_4]:
                time_scale = 4.0

            # Optional: focus camera on a random live fish when pressing space
            if keys[pygame.K_SPACE] and world.fish:
                focus = random.choice(world.fish)
                camera_x, camera_y = focus.x, focus.y

            # Clamp camera so it stays over the real tank
            visible_half_w = VIEW_WIDTH / (2 * zoom)
            visible_half_h = VIEW_HEIGHT / (2 * zoom)
            camera_x = clamp(camera_x, visible_half_w, WORLD_WIDTH - visible_half_w)
            camera_y = clamp(camera_y, visible_half_h, WORLD_HEIGHT - visible_half_h)

            sim_dt = dt * time_scale
            sim_time += sim_dt

            eaten = world.update(sim_dt, sim_time)
            if sim_dt > 0:
                rate = eaten / sim_dt
                consumption_history.append((sim_time, rate))
                # Keep only last ~10 seconds + small buffer
                cutoff = sim_time - 11.0
                consumption_history = [(t, r) for (t, r) in consumption_history if t >= cutoff]

            draw_world(
                screen,
                world,
                (camera_x, camera_y),
                zoom,
                show_graph,
                consumption_history,
                sim_time,
                time_scale,
            )
        else:
            draw_menu(screen, menu_options, menu_index, menu_message, world is not None)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()


