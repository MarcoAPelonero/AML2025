import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt

class LinearAgent:
    """
    A simple linear agent for reinforcement learning using a softmax policy. 
    The agent maps input states to action probabilities using a single weight matrix.

    Attributes:
        input_dim (int): Size of the input state vector.
        output_dim (int): Number of possible actions.
        learning_rate (float): Step size for weight updates.
        temperature (float): Controls exploration in softmax policy.
        weights (np.ndarray): Action weights matrix of shape (output_dim, input_dim).
        gradients (list): Stores accumulated gradients for batch updates.

    Methods:
        forward(state): Computes action logits from input state.
        policy(state): Returns action probabilities using softmax.
        sample_action(state): Samples an action and returns adjusted probabilities.
        update_weights(state, action, reward): Updates weights using REINFORCE.
        accumulate_gradients(state, action, reward): Stores gradients for batch update.
        apply_gradients(): Applies accumulated gradients to weights.
        reset_parameters(): Resets weights and gradients.
        render_weights(): Visualizes weights as a 3D bar plot.
    """
    def __init__(self, input_dim = 25, output_dim = 4, learning_rate=0.02, temperature=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gradients = []

        self.weights = np.zeros((output_dim, input_dim))

    def forward(self, state):
        return np.dot(self.weights, state)

    def policy(self, state):
        logits = self.forward(state) / self.temperature
        exps = np.exp(logits - np.max(logits))  
        probs = exps / np.sum(exps)
        return probs

    def sample_action(self, state):
        probs = self.policy(state)
        action = np.random.choice(self.output_dim, p=probs)
        probs[action] -= 1.0  
        return action, probs

    def update_weights(self, state, action, reward):
        """
        Update weights and bias using REINFORCE gradient:
        ∇θ log π(a|s) * R
        """
        probs = self.policy(state)
        probs[action] -= 1.0

        grad_weights = np.outer(probs, state)
        dw_out = np.copy(-self.learning_rate * reward * grad_weights)
        self.weights += dw_out

        return dw_out.copy()

    def accumulate_gradients(self, state, action, reward):
        probs = self.policy(state)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0

        grad_logits  = one_hot - probs
        grad_weights = np.outer(grad_logits, state)

        self.gradients.append(reward * grad_weights)

    def apply_gradients(self):
        if not self.gradients:
            return
        
        weights_array = np.stack([gw for gw in self.gradients])
        total_grad_weights = np.sum(weights_array, axis=0).copy()

        self.weights += self.learning_rate * total_grad_weights
        self.gradients.clear()

    def reset_parameters(self):
        self.weights = np.zeros((self.output_dim, self.input_dim))
        self.gradients.clear()

    def render_weights(self):
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(self.input_dim)
        y = np.arange(self.output_dim)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)

        dx = dy = 0.8
        dz = self.weights[y, x]

        ax.bar3d(x, y, z, dx, dy, dz, color='skyblue', alpha=0.8)
        ax.set_xlabel('Input Index')
        ax.set_ylabel('Action')
        ax.set_zlabel('Weight Value')
        ax.set_title('Agent Weights per Action and Input')

        plt.show()

def test_agent():
    """
    Tests all the functionalities of the LinearAgent class, whilst interacting with the Environment class.
    """

    # Test agent initialization and basic methods
    spatial_res = 5
    input_dim = spatial_res**2
    output_dim = 4 

    agent = LinearAgent(input_dim, output_dim)
    state = np.random.rand(input_dim) 
    probs = agent.policy(state)
    print("Action Probabilities:", probs)
    action = agent.sample_action(state)

    # agent.render_weights()

    from environment import Environment

    # Test agent-environment interaction
    env = Environment(grid_size=spatial_res, sigma=0.2)
    env.reset()
    
    agent_position = env.encoded_position
    action_probs = agent.policy(agent_position.flatten())
    print("Encoded Agent Position:", agent_position)
    print("Action Probabilities:", action_probs)

    path = []
    env.reset()
    done = False

    for step in range(1000):
        agent_position = env.encoded_position
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        path.append(env.agent_position.copy())
        if done:
            print(f"Episode finished after {step + 1} steps with total reward: {reward}")
            break

    counter = 0
    for episodes in range(1000):
        env.reset()
        done = False
        time = 0
        while not done:
            time += 1
            if time > 30:
                break
            agent_position = env.encoded_position
            action = agent.sample_action(agent_position.flatten())
            reward, done = env.step(action)
        if done:
            counter += 1
    print(f"Done episodes (pre training): {counter} out of 1000")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)

    colors = cm.viridis(np.linspace(0, 1, len(path)))

    for i in range(len(path) - 1):
        ax.plot(
            [path[i][0], path[i + 1][0]],
            [path[i][1], path[i + 1][1]],
            color=colors[i],
            marker='o'
        )

    ax.plot(env.food_position[0], env.food_position[1], 'rx', markersize=10, label='Food')

    ax.set_title('Agent Path and Food Position')
    ax.legend()
    plt.show()

    print("Total Reward:", reward)

    agent.reset_parameters()
    path = []
    env.reset()
    done = False
    agent.learning_rate = 0.1  
    
    # Single episode training and check the weight shift after training

    while not done:
        agent_position = env.encoded_position
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        agent.update_weights(agent_position.flatten(), action, reward)
        path.append(env.agent_position.copy())

    agent.render_weights()
    agent_position = env.encode_position((0, 0))
    action_probs = agent.policy(agent_position.flatten())
    print("Action Probabilities after training at (0, 0):", action_probs)

    counter = 0
    for episodes in range(1000):
        env.reset()
        done = False
        time = 0
        while not done:
            time += 1
            if time > 30:
                break
            agent_position = env.encoded_position
            action = agent.sample_action(agent_position.flatten())
            reward, done = env.step(action)
        if done:
            counter += 1
    print(f"Done episodes (post training): {counter} out of 1000")

    agent = LinearAgent(input_dim, output_dim, learning_rate=0.01, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    env.reset()
    agent.reset_parameters()

    episodes = 600
    reward_list = []
    for episode in range(episodes):
        env.reset()
        done = False
        time = 0
        while not done:
            time += 1
            if time > 30:
                break
            agent_position = env.encoded_position
            action = agent.sample_action(agent_position.flatten())
            reward, done = env.step(action)
            agent.update_weights(agent_position.flatten(), action, reward)

        reward_list.append(reward)  
    
    agent_position = env.encode_position((0, 0))
    action_probs = agent.policy(agent_position.flatten())
    print("Action Probabilities after training at (0, 0):", action_probs)
    agent.render_weights()
    reward_list = [np.mean(reward_list[max(0, i-20):i+1]) for i in range(len(reward_list))]
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()

    counter = 0
    for episodes in range(1000):
        env.reset()
        done = False
        time = 0
        while not done:
            time += 1
            if time > 30:
                break
            agent_position = env.encoded_position
            action = agent.sample_action(agent_position.flatten())
            reward, done = env.step(action)
        if done:
            counter += 1
    print(f"Done episodes (final test): {counter} out of 1000")
    
    agent = LinearAgent(input_dim, output_dim, learning_rate=0.01, temperature=1.0)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    env.reset()
    agent.reset_parameters()

    episodes = 600
    reward_list = []
    for episode in range(episodes):
        env.reset()
        done = False
        time = 0
        while not done:
            time += 1
            if time > 30:
                break
            agent_position = env.encoded_position
            action = agent.sample_action(agent_position.flatten())
            reward, done = env.step(action)
            agent.accumulate_gradients(agent_position.flatten(), action, reward)
        agent.apply_gradients()

        reward_list.append(reward)  
    
    agent_position = env.encode_position((0, 0))
    action_probs = agent.policy(agent_position.flatten())
    print("Action Probabilities after training at (0, 0):", action_probs)
    agent.render_weights()
    reward_list = [np.mean(reward_list[max(0, i-20):i+1]) for i in range(len(reward_list))]
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()

if __name__ == "__main__":
    test_agent()