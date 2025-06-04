import pygame
import torch
import numpy as np
import argparse
from agent import REINFORCEAgent  # see [agent.py](agent.py)
from game import BreakoutPongEnv  # see [game.py](game.py)

def main(policy_file):
    episodes = 5
    env = BreakoutPongEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent and load trained weights
    agent = REINFORCEAgent(state_dim, action_dim, num_envs=1)
    agent.load(policy_file)

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            # Allow window events (e.g. closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            with torch.no_grad():
                action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

        # Clear agent buffers after each episode
        agent.finish_rollout()

        print(f"Episode {ep+1} finished.")
        pygame.time.delay(1000)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pong visualizer with a specified policy file.")
    parser.add_argument("--policy", type=str, default="policy.pth", help="Path to the policy file")
    args = parser.parse_args()
    main(args.policy)