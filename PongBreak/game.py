from __future__ import annotations
import math
from typing import Any, Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class BreakoutPongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps

        self.screen_width = 400
        self.screen_height = 600

        self.paddle_width = 80
        self.paddle_height = 10
        self.paddle_max_speed = 10.0
        self.paddle_acc = 1.0
        self.paddle_friction = 0.9

        self.ball_radius = 5
        self.ball_init_speed_y = -5.0
        self.ball_max_speed = 7.0
        self.ball_speed_gain = 0.1

        self.brick_rows = 3
        self.brick_cols = 10
        self.brick_width = self.screen_width // self.brick_cols
        self.brick_height = 20
        self.brick_offset_y = 50

        self.action_space = spaces.Discrete(3)  # 0-idle | 1-left | 2-right

        low = np.array([0.0, 0.0, -7.0, -7.0, 0.0, -10.0], dtype=np.float32)
        high = np.array([400.0, 600.0, 7.0, 7.0, 320.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._step_counter: int = 0
        self.ball_x: float
        self.ball_y: float
        self.ball_vx: float
        self.ball_vy: float
        self.paddle_x: float
        self.paddle_vx: float
        self.bricks: np.ndarray

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_counter = 0

        self.paddle_x = (self.screen_width - self.paddle_width) / 2.0
        self.paddle_vx = 0.0

        self.ball_x = self.screen_width / 2.0
        self.ball_y = self.screen_height / 2.0
        self.ball_vx = self.np_random.uniform(-3.0, 3.0)
        self.ball_vx = np.clip(self.ball_vx, -self.ball_max_speed, self.ball_max_speed)
        self.ball_vy = self.ball_init_speed_y

        self.bricks = np.ones((self.brick_rows, self.brick_cols), dtype=bool)

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_counter += 1
        terminated = truncated = False
        reward = 0.0

        if action == 1:
            self.paddle_vx -= self.paddle_acc
        elif action == 2:
            self.paddle_vx += self.paddle_acc

        self.paddle_vx *= self.paddle_friction
        self.paddle_vx = np.clip(self.paddle_vx, -self.paddle_max_speed, self.paddle_max_speed)
        self.paddle_x = np.clip(self.paddle_x + self.paddle_vx, 0.0, self.screen_width - self.paddle_width)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x - self.ball_radius <= 0:
            self.ball_x = self.ball_radius
            self.ball_vx = -self.ball_vx
        elif self.ball_x + self.ball_radius >= self.screen_width:
            self.ball_x = self.screen_width - self.ball_radius
            self.ball_vx = -self.ball_vx

        if self.ball_y - self.ball_radius <= 0:
            self.ball_y = self.ball_radius
            self.ball_vy = -self.ball_vy

        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if not self.bricks[r, c]:
                    continue
                bx = c * self.brick_width
                by = self.brick_offset_y + r * self.brick_height
                if bx <= self.ball_x <= bx + self.brick_width and \
                   by <= self.ball_y <= by + self.brick_height:
                    self.bricks[r, c] = False
                    self.ball_vy = -self.ball_vy
                    self._accelerate_ball()
                    break
            else:
                continue
            break

        paddle_top = self.screen_height - self.paddle_height
        if (self.ball_y + self.ball_radius >= paddle_top and
            self.paddle_x <= self.ball_x <= self.paddle_x + self.paddle_width and
            self.ball_vy > 0):
            self.ball_y = paddle_top - self.ball_radius
            self.ball_vy = -self.ball_vy
            offset = (self.ball_x - (self.paddle_x + self.paddle_width / 2)) / (self.paddle_width / 2)
            self.ball_vx += offset * 1.5
            self._accelerate_ball()
            reward += 0.5

        if self.ball_y - self.ball_radius > self.screen_height:
            terminated = True
            reward = - 2

        if not self.bricks.any():
            terminated = True

        if self._max_episode_steps and self._step_counter >= self._max_episode_steps:
            truncated = True

        if not terminated and not truncated:
            paddle_cx = self.paddle_x + self.paddle_width / 2.0
            dist = abs(self.ball_x - paddle_cx)
            dist_norm = dist / (self.screen_width / 2.0)
            shaping = max(0.0, 1.0 - dist_norm) * 0.15
            reward += shaping

        if self.render_mode == "human":
            self._draw_frame()

        return self._get_obs(), reward, terminated, truncated, {}

    def _accelerate_ball(self) -> None:
        self.ball_vx = float(np.clip(self.ball_vx * (1 + self.ball_speed_gain / 10),
                                     -self.ball_max_speed, self.ball_max_speed))
        self.ball_vy = float(np.clip(self.ball_vy * (1 + self.ball_speed_gain / 10),
                                     -self.ball_max_speed, self.ball_max_speed))

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_x, self.paddle_vx],
            dtype=np.float32,
        )

    def _init_pygame(self) -> None:
        if self._screen is not None:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Breakout-Pong Env")
        self._clock = pygame.time.Clock()

    def _draw_frame(self) -> None:
        assert self._screen is not None and self._clock is not None
        self._screen.fill((0, 0, 0))

        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if not self.bricks[r, c]:
                    continue
                rect = pygame.Rect(
                    c * self.brick_width,
                    self.brick_offset_y + r * self.brick_height,
                    self.brick_width,
                    self.brick_height,
                )
                pygame.draw.rect(self._screen, (255, 0, 0), rect)
                pygame.draw.rect(self._screen, (0, 0, 0), rect, 1)

        paddle_rect = pygame.Rect(
            int(self.paddle_x),
            self.screen_height - self.paddle_height,
            self.paddle_width,
            self.paddle_height,
        )
        pygame.draw.rect(self._screen, (255, 255, 255), paddle_rect)

        pygame.draw.circle(self._screen, (255, 0, 0),
                           (int(self.ball_x), int(self.ball_y)), self.ball_radius)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def render(self) -> None | np.ndarray:
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            if self._screen is None:
                self._init_pygame()
            self._draw_frame()
            return np.transpose(pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen, self._clock = None, None


if __name__ == "__main__":
    env = BreakoutPongEnv(render_mode="human")
    obs, _ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_LEFT] else 2 if keys[pygame.K_RIGHT] else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print("Episode finished:", "won!" if reward > 0 else "lost!")
            pygame.time.delay(1000)
            running = False

    env.close()
