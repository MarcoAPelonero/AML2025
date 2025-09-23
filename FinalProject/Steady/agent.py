import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt

class LinearAgent:
    """
    A softmax policy agent with a single linear readout (no hidden layers).

    The agent computes action logits as a linear function of the input state,
    converts them to a categorical distribution via softmax (optionally
    temperature-scaled), samples actions, and updates its weights with a
    REINFORCE-style policy gradient.

    - Policy: π(a|s) = softmax( (W s) / T ) where W ∈ R^{A×D}, s ∈ R^{D},
      T > 0 is the temperature.
    - Single-step REINFORCE update (no baseline):
      ΔW = α · R · ∇_W log π(a|s) with
      ∇_W log π(a|s) = (one_hot(a) − π(·|s)) x s
      where x denotes outer product.
    - Two update modes are provided:
        1) `update_weights`: immediate online update per (s, a, R).
        2) `accumulate_gradients` + `apply_gradients`: accumulate per-step
           gradients (scaled by rewards) and apply them in a single batch step.
    """
    def __init__(self, input_dim = 25, output_dim = 4, learning_rate=0.02, temperature=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.gradients = []

        self.weights = np.zeros((output_dim, input_dim))

    def forward(self, state):
        """
        Compute action logits from the input state.
        """
        return np.dot(self.weights, state)

    def policy(self, state):
        """
        Compute action probabilities via softmax policy.
        """
        logits = self.forward(state) / self.temperature
        exps = np.exp(logits - np.max(logits))  
        probs = exps / np.sum(exps)
        return probs

    def sample_action(self, state):
        """
        Sample an action from the policy distribution.
        """
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
        """
        Accumulate gradients for a single (state, action, reward) tuple.
        """
        probs = self.policy(state)
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0

        grad_logits  = one_hot - probs
        grad_weights = np.outer(grad_logits, state)

        self.gradients.append(reward * grad_weights)

    def apply_gradients(self):
        """
        Apply accumulated gradients to update weights.
        """
        if not self.gradients:
            return
        
        weights_array = np.stack([gw for gw in self.gradients])
        total_grad_weights = np.sum(weights_array, axis=0).copy()

        self.weights += self.learning_rate * total_grad_weights
        self.gradients.clear()

    def reset_parameters(self):
        """
        Reset weights and clear accumulated gradients.
        """
        self.weights = np.zeros((self.output_dim, self.input_dim))
        self.gradients.clear()

    def render_weights(self, show: bool = True, return_fig: bool = False, ax=None):
        """
        Visualize the agent's weights as a 3D bar plot.

        Args:
            show (bool): If True, calls plt.show(). If False, does not display.
            return_fig (bool): If True, returns (fig, ax) for further use.
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D or None): Optional existing 3D axis.

        Returns:
            (fig, ax) if return_fig=True, else None.
        """
        from mpl_toolkits.mplot3d import Axes3D  

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = True
        else:
            fig = ax.figure

        input_dim = self.input_dim
        output_dim = self.output_dim

        x = np.arange(input_dim)
        y = np.arange(output_dim)
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        Z0 = np.zeros_like(X)

        dx = dy = 0.8
        dz = self.weights[Y, X]

        ax.bar3d(X, Y, Z0, dx, dy, dz, alpha=0.85)
        ax.set_xlabel('Input Index')
        ax.set_ylabel('Action')
        ax.set_zlabel('Weight Value')
        ax.set_title('Agent Weights per Action and Input')
        ax.set_xlim(-0.5, input_dim - 0.5)
        ax.set_ylim(-0.5, output_dim - 0.5)

        if show:
            plt.show()

        if return_fig:
            return fig, ax

        if (not show) and created_fig:
            plt.close(fig)
        return None
    
def animate_weights(weights_over_time: np.ndarray,
                    interval: int = 40,
                    save_path: str | None = 'weight_anim.gif',
                    dpi: int = 120,
                    frame_skip: int = 10,
                    fig=None,
                    ax=None) -> tuple:
    """
    Create an animation of evolving weights over episodes as a 3D bar plot.

    Args:
        weights_over_time (np.ndarray): shape (episodes, output_dim, input_dim)
        interval (int): Delay between frames in milliseconds.
        save_path (str|None): If provided, saves animation to this path.
                              Supports '.mp4' (requires ffmpeg) or '.gif'.
        dpi (int): DPI when saving to file.
        frame_skip (int): Use every nth frame to speed up the animation.
        fig: Optional matplotlib.figure.Figure to draw into.
        ax: Optional 3D Axes to draw into. If provided, must be projection='3d'.

    Returns:
        anim, fig: (matplotlib.animation.FuncAnimation, matplotlib.figure.Figure)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from pathlib import Path

    if weights_over_time.ndim != 3:
        raise ValueError("weights_over_time must be (episodes, output_dim, input_dim)")

    n_frames, output_dim, input_dim = weights_over_time.shape
    if n_frames == 0:
        raise ValueError("weights_over_time must contain at least one frame")

    if frame_skip <= 0:
        raise ValueError("frame_skip must be a positive integer")

    frame_indices = np.arange(0, n_frames, frame_skip, dtype=int)
    frame_indices = np.unique(np.append(frame_indices, [0, n_frames - 1]))

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        if fig is None:
            fig = ax.figure
        if not hasattr(ax, "bar3d"):
            raise TypeError("ax must be a 3D axis (projection='3d').")

    x = np.arange(input_dim)
    y = np.arange(output_dim)
    X, Y = np.meshgrid(x, y)
    Xf = X.flatten()
    Yf = Y.flatten()
    Z0 = np.zeros_like(Xf)
    dx = dy = 0.8

    max_abs = np.nanmax(np.abs(weights_over_time))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    def _style_ax():
        ax.grid(False)
        ax.set_axis_off()
        ax.set_facecolor((0, 0, 0, 0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_visible(False)
                axis._axinfo["grid"]['linewidth'] = 0
            except Exception:
                pass

    def _draw_frame(frame_idx: int):
        ax.clear()
        W = weights_over_time[frame_idx]
        dz = W[Yf, Xf]

        ax.bar3d(Xf, Yf, Z0, dx, dy, dz, alpha=0.85)

        ax.set_xlim(-0.5, input_dim - 0.5)
        ax.set_ylim(-0.5, output_dim - 0.5)
        ax.set_zlim(-max_abs, max_abs)

        _style_ax()
        ax.set_title(f'Agent Weights per Action and Input - Episode {frame_idx + 1}/{n_frames}')

    def init():
        _draw_frame(frame_indices[0])
        return []

    def update(frame_idx):
        _draw_frame(frame_idx)
        return []

    anim = FuncAnimation(fig, update, frames=frame_indices, init_func=init,
                         interval=interval, blit=False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        effective_interval = max(1, interval)
        fps = max(1, int(1000 / effective_interval))
        suffix = save_path.suffix.lower()
        if suffix == '.mp4':
            writer = FFMpegWriter(fps=fps, bitrate=1800)
        elif suffix == '.gif':
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError("save_path must end with .mp4 or .gif")
        anim.save(save_path, writer=writer, dpi=dpi)

    return anim, fig



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