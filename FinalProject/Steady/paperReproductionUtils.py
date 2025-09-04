from trainingUtils import OutOfDistributionTraining
from reservoirTrainingUtils import InDistributionTraining, InDistributionInference, OODInference
from plottingUtils import agg, plot_trajectories
from agent import LinearAgent
from environment import Environment
from plottingUtils import plot_rewards_as_article, plot_trajectories_ood, plot_out_of_distribution_comparison
import numpy as np
import matplotlib.pyplot as plt

def plotOutOfDistributionResultsGradOnly():
    from environment import Environment
    from agent import LinearAgent  

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.02, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    rewards, trajectories = OutOfDistributionTraining(agent, env, rounds=1, episodes=600, time_steps=30, mode='normal', verbose=False)
    print(rewards.shape)
    print(trajectories[0]['trajectory'].shape)
    print("Training complete.")

    plot_rewards_as_article(rewards, savefig=True, filename="presentation_figures/rewards_plot.png")
    plot_trajectories_ood(trajectories, batch_size = 200, savefig=True, filename="presentation_figures/trajectories_plot.png")

def plotOutOfDistributionResultsReservoir():
    from environment import Environment
    from agent import LinearAgent  
    from reservoir import initialize_reservoir, build_W_out

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.02, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)

    reservoir = initialize_reservoir()

    rewards, trajectories, reservoir_states, gradients = InDistributionTraining(agent, env, reservoir, rounds=3, episodes=600, verbose=True)

    rewards_grad, _ = OutOfDistributionTraining(agent, env, rounds=1, episodes=600, time_steps=30, mode='normal', verbose=False)

    rewards_grad_even = rewards_grad[::2]
    rewards_grad_odd = rewards_grad[1::2]

    res_states = reservoir_states.reshape(-1, reservoir_states.shape[-1])
    grads = gradients.reshape(-1, gradients.shape[-1])
    res_states = res_states[~np.isnan(res_states).any(axis=1)]
    grads = grads[~np.isnan(grads).any(axis=1)]
    print("Res States Shape:", res_states.shape)
    print("Gradients Shape:", grads.shape)

    W_out = build_W_out(res_states, grads)
    reservoir.Jout = W_out.T

    rewards,trajectories,_,_ = OODInference(agent, env, reservoir, rounds=1, episodes=600, verbose=True)

    rewards_even = rewards[::2]
    rewards_odd = rewards[1::2]

    rewards_grad_even = np.array(rewards_grad_even)
    rewards_grad_odd = np.array(rewards_grad_odd)

    rewards_even = np.array(rewards_even)
    rewards_odd = np.array(rewards_odd)

    print("Rewards grad even shape:", rewards_grad_even.shape)
    print("Rewards grad odd shape:", rewards_grad_odd.shape)
    print("Rewards even shape:", rewards_even.shape)
    print("Rewards odd shape:", rewards_odd.shape)
# jdf
    plot_out_of_distribution_comparison(
        rewards_grad_even, rewards_grad_odd,
        rewards_even, rewards_odd,
        bin_size=25, high_point=1.5,
        figsize=(12, 10), savefig=True, filename="presentation_figures/ood_comparison.png"
    )

if __name__ == "__main__":
    plotOutOfDistributionResultsReservoir()