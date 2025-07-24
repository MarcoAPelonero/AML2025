import numpy as np

class ReservoirESN:
    """
    Echo State Network with optional multiplicative modulation.
    
    If modulation_dim > 0, then at each step:
        gain = Wmod @ m
        pre  = (W @ x + Win @ u + bias) * (1 + gain)
    otherwise standard ESN update.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 500,
        output_dim: int = 1,
        spectral_radius: float = 0.90,
        sparsity: float = 0.8,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
        modulation_dim: int = 0,
        mod_scaling: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.leak_rate = leak_rate
        self.modulation_dim = modulation_dim

        rng = np.random.default_rng(seed)

        # Input weights
        self.Win = rng.uniform(-1, 1, (reservoir_size, input_dim)) * input_scaling
        # Optional modulation weights
        if modulation_dim > 0:
            self.Wmod = rng.uniform(-1, 1, (reservoir_size, modulation_dim)) * mod_scaling
        else:
            self.Wmod = None

        # Reservoir recurrent weights
        W = rng.uniform(-0.5, 0.5, (reservoir_size, reservoir_size))
        mask = rng.random(W.shape) < sparsity
        W[mask] = 0.0
        # scale to spectral radius
        eigs = np.linalg.eigvals(W)
        W *= (spectral_radius / np.max(np.abs(eigs)))
        self.W = W

        # Read‑out
        self.Wout = rng.uniform(-0.1, 0.1, (output_dim, reservoir_size))
        self.bout = np.zeros(output_dim)

        # bias inside reservoir (often helpful)
        self.b = np.zeros(reservoir_size)

        self.reset_state()

    def reset_state(self) -> None:
        self.activity = []
        self.x = np.zeros(self.reservoir_size)

    def update(self, u: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
        #--- standard pre-activation
        pre = self.W @ self.x + self.Win @ u + self.b

        #--- optional multiplicative modulation
        if self.Wmod is not None:
            if m is None or m.shape[0] != self.modulation_dim:
                raise ValueError(f"Expected modulation vector of shape ({self.modulation_dim},), got {m!r}")
            gain = self.Wmod @ m             # real-valued, continuous
            pre = pre * (1.0 + gain)        # safe: if gain=0, no change

        #--- leaky integration + tanh
        x_new = (1 - self.leak_rate) * self.x + self.leak_rate * np.tanh(pre)
        self.x = x_new
        self.activity.append(self.x.copy())
        return self.x

    def readout(self, activation: str | None = None) -> np.ndarray:
        y = self.Wout @ self.x + self.bout
        if activation is None:
            return y
        if activation == "tanh":
            return np.tanh(y)
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-y))
        raise ValueError(f"Unknown activation '{activation}'")

    def step(self, u: np.ndarray, m: np.ndarray | None = None, *, activation: str | None = None) -> np.ndarray:
        self.update(u, m)
        return self.readout(activation)
    
    def return_activity(self) -> np.ndarray:
        return np.array(self.activity)

def initialize_reservoir(agent, environment, reservoir_size=500, spectral_radius=0.95,
                         sparsity=0.8, input_scaling=1.0, leak_rate=0.8,
                         modulation_dim=5, mod_scaling=1.0, seed=None):
    
    input_dim = environment.encoded_position.shape[0] + environment.encode(0, res=5).shape[0]  # Assuming action encoding is similar to position encoding
    output_dim = agent.weights.size + agent.bias.size  # Total number of parameters in the

    reservoir = ReservoirESN(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        output_dim=output_dim,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leak_rate=leak_rate,
        modulation_dim=modulation_dim,
        mod_scaling=mod_scaling,
        seed=seed
    )

    return reservoir

def testing():
    
    from environment import Environment
    from agent import LinearAgent

    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4

    agent = LinearAgent(input_dim, output_dim)
    env = Environment(grid_size=spatial_res, sigma=0.2)

    # Now we simulate one step of the environment, and we check the output dimension
    env.reset_inner()
    agent_position = env.encoded_position
    action = agent.sample_action(agent_position.flatten())
    reward, done = env.step(action)
    weight_update, bias_update = agent.update_weights(agent_position.flatten(), action, reward)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

    encoded_action = env.encode(action, res=5)

    enc0 = env.encode(0, res=5)
    print(f"Encoded 0: {enc0}, Shape: {enc0.shape}")
    enc1 = env.encode(1, res=5)
    print(f"Encoded 1: {enc1}, Shape: {enc1.shape}")
    enc2 = env.encode(2, res=5)
    print(f"Encoded 2: {enc2}, Shape: {enc2.shape}")
    enc3 = env.encode(3, res=5)
    print(f"Encoded 3: {enc3}, Shape: {enc3.shape}")

    print(f"Encoded Position: {agent_position}, Shape: {agent_position.shape}")
    print(f"Weight Update: {weight_update.shape}, Bias Update: {bias_update.shape}")

    reservoir = initialize_reservoir(agent, env, reservoir_size=100)
    print(f"Reservoir initialized with input_dim: {reservoir.input_dim}, reservoir_size: {reservoir.reservoir_size}, output_dim: {reservoir.output_dim}")

    # Now simulate an episode of 30 steps without trajectory accumulation
    time_steps = 30
    time = 0
    done = False

    reservoir.reset_state()
    res_states = []
    traj = []

    while time < time_steps and not done:
        agent_position = env.encoded_position
        traj.append(env.agent_position)
        action = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)
        weight_update, bias_update = agent.update_weights(agent_position.flatten(), action, reward)
        encoded_action = env.encode(action, res=5)
        reservoir_input = np.concatenate((agent_position.flatten(), encoded_action.flatten()))
        reservoir_output = reservoir.update(reservoir_input, m=env.encode(reward, res=5))
        res_states.append(reservoir_output.copy())
        time += 1

    res_states = np.array(res_states)
    print(f"Reservoir states shape: {res_states.shape}")
    import matplotlib.pyplot as plt
    T, N = res_states.shape
    time = np.arange(T)

    # pick 500 distinct colors from a colormap
    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, N))

    plt.figure(figsize=(12, 6))
    for i in range(N):
        plt.plot(time, res_states[:, i], color=colors[i], linewidth=0.7)

    plt.xlabel('Time step')
    plt.ylabel('Activation')
    plt.title('Reservoir Neuron Activities (500 lines)')
    plt.tight_layout()
    plt.show()

    # Now plot that same activity, but make it a subplot and in the next subplot plot the trajectory as lines that connect each past postion to the next
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i in range(N):
        plt.plot(time, res_states[:, i], color=colors[i], linewidth=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Activation')
    plt.title('Reservoir Neuron Activities (500 lines)')
    plt.subplot(1, 2, 2)
    for i in range(len(traj) - 1):
        plt.plot([traj[i][0], traj[i + 1][0]], [traj[i][1], traj[i + 1][1]], color='gray', linewidth=0.5)
    plt.xlim(-0.75,0.75)
    plt.ylim(-0.75,0.75)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectory')
    plt.tight_layout()
    plt.show()

def testing_various_paths():
    from environment import Environment
    from agent import LinearAgent
    import numpy as np
    import matplotlib.pyplot as plt
    spatial_res = 5
    input_dim = spatial_res ** 2
    output_dim = 4
    agent = LinearAgent(input_dim, output_dim)
    env = Environment(grid_size=spatial_res, sigma=0.2)
    reservoir = initialize_reservoir(agent, env, reservoir_size=100, spectral_radius=0.9)

    time_steps = 30
    n_runs = 8

    # create 4x4 grid: each run -> [traj, activity]
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    axs = axs.flatten()

    for run in range(n_runs):
        # reset per‐run
        env.reset()
        reservoir.reset_state()
        traj = []

        # simulate
        for t in range(time_steps):
            pos_encoded = env.encoded_position.flatten()
            traj.append(env.agent_position.copy())
            action = agent.sample_action(pos_encoded)
            reward, done = env.step(action)

            inp = np.concatenate((pos_encoded, env.encode(action, res=5).flatten()))
            _ = reservoir.update(inp, m=env.encode(reward, res=5))

        # plot trajectory
        ax_traj = axs[2 * run]
        traj = np.array(traj)
        ax_traj.plot(traj[:, 0], traj[:, 1], '-o', markersize=3)
        ax_traj.set_title(f'Run {run + 1} Traj')
        ax_traj.set_xlim(-0.75, 0.75)
        ax_traj.set_ylim(-0.75, 0.75)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.grid(False)

        # plot reservoir activity
        ax_act = axs[2 * run + 1]
        activity = reservoir.return_activity()  # shape: (reservoir_size,)
        ax_act.plot(activity)
        ax_act.set_title(f'Run {run + 1} Activity')
        ax_act.set_xlabel('Neuron index')
        ax_act.set_ylabel('Activation')
        ax_act.grid(False)

    # hide any unused axes (in case 16 > 2*n_runs)
    for ax in axs[2 * n_runs:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def test_final_state_without_paths(
        spatial_res: int = 5,
        time_steps: int = 30,
        n_runs: int = 100,           # raise if you want more samples
        distance_metric: str = "cosine"
    ):
    """
    Run `n_runs` roll‑outs, collect the reservoir’s final activation vector
    for each run, and visualise:
      1. A 2‑D PCA scatter of those vectors.
      2. A pair‑wise distance heat‑map (using `distance_metric`).
    
    Returns
    -------
    final_states : ndarray, shape (n_runs, reservoir_size)
        Each row is the final reservoir activation for one run.
    D            : ndarray, shape (n_runs, n_runs)
        Distance matrix (square‑form) between final states.
    pts          : ndarray, shape (n_runs, 2)
        The 2‑D PCA coordinates used in the scatter plot.
    """
    # --- imports ------------------------------------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist, squareform

    from environment import Environment
    from agent import LinearAgent
    # -----------------------------------------------------------------------

    # --- build agent, env, reservoir ---------------------------------------
    input_dim  = spatial_res ** 2
    output_dim = 4
    agent      = LinearAgent(input_dim, output_dim)
    env        = Environment(grid_size=spatial_res, sigma=0.2)
    reservoir = initialize_reservoir(agent, env, reservoir_size=1000, spectral_radius=0.9,
                         sparsity=0.9, input_scaling=3.0, leak_rate=1,
                         modulation_dim=5, mod_scaling=1.0, seed=None)
    # -----------------------------------------------------------------------

    final_states = []

    # ───── Simulate ─────────────────────────────────────────────────────────
    for _ in range(n_runs):
        env.reset()
        reservoir.reset_state()
        probs = np.random.rand(4)
        probs = probs / np.sum(probs) 

        # Now make prob as a 1 hot vector with 4 zeros and 1 1
        # probs = np.zeros(4)
        # probs[np.random.choice(4)] = 1

        for _ in range(time_steps):
            pos_enc = env.encoded_position.flatten()
            action = np.random.choice(4, p=probs)  # Sample action based on probabilities
            reward, _ = env.step(action)

            inp = np.concatenate((pos_enc,
                                   env.encode(action, res=spatial_res).flatten()))
            reservoir.update(inp, m=env.encode(reward, res=spatial_res))

        final_states.append(reservoir.x.copy())

    final_states = np.vstack(final_states)  # (n_runs, reservoir_size)

    # ───── 1) PCA scatter ──────────────────────────────────────────────────
    pca = PCA(n_components=2)
    pts = pca.fit_transform(final_states)
    D = squareform(pdist(final_states, metric=distance_metric))

    # Now plot both of them in one single plot with 2 subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pts[:, 0], pts[:, 1], s=60)
    for i, (x, y) in enumerate(pts):
        plt.text(x, y, str(i + 1), ha='center', va='center', weight='bold')
    plt.title('Final reservoir state (PCA projection)')
    plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.grid(True)
    plt.subplot(1, 2, 2)
    im = plt.imshow(D, cmap='viridis')
    plt.colorbar(im, label=f'{distance_metric.title()} distance')
    plt.title('Pair‑wise distance between final states')
    plt.xlabel('Run'); plt.ylabel('Run')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_final_state_without_paths()
    testing_various_paths()