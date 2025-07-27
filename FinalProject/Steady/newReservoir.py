import numpy as np

class Reservoir:
    def __init__(self):
        self.N = 500
        self.I = 25 + 4 + 10  
        self.O = 4 * 25 + 4     
        self.T = 30
        self.dt = 0.3

        tau_m_f = 3.00 * self.dt
        tau_m_s = 100.00 * self.dt
        tau_s = 2. * self.dt
        tau_ro = 0.001 * self.dt
        self.tau_m = np.linspace(tau_m_f, tau_m_s, self.N)
        self.itau_s = np.exp(-self.dt / tau_s)
        self.itau_ro = np.exp(-self.dt / tau_ro)

        self.dv = 5.0
        self.Vo = 0.0
        h = 0.0
        self.h = np.ones(self.N) * h

        sigma_input = 0.5
        sigma_rec = 0.5 / np.sqrt(self.N)
        sigma_teach = 0.0
        sigma_output = 0.1

        self.J = np.random.normal(0.0, sigma_rec, size=(self.N, self.N))
        self.Jin = np.random.normal(0.0, sigma_input, size=(self.N, self.I))
        self.Jteach = np.random.normal(0.0, sigma_teach, size=(self.N, self.O))
        self.Jout = np.random.normal(0.0, sigma_output, size=(self.O, self.N))
        self.Jin_mult = np.random.normal(0,0.1,size=(self.N,))
        self.h_Jout = np.zeros((self.O,))

        s_inh = 0.0
        self.s_inh = -s_inh
        self.Jreset = np.diag(np.ones(self.N) * self.s_inh)

        self.S = np.zeros(self.N)
        self.S_hat = np.zeros(self.N)
        self.S_ro = np.zeros(self.N)
        self.H = np.ones(self.N) * self.Vo * 0.0
        self.y = np.zeros(self.O)

        self.state_out = np.zeros(self.N)
        self.state_out_p = np.zeros(self.N)

        self.state_trace = []

    def step_rate (self, inp, inp_modulation,sigma_S=0.00, if_tanh=True):
        self.S_hat   = np.copy(self.S)
        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh( ( self.J @ self.S_hat   + self.Jin @ inp ) * (inp_modulation) )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))
        self.S_ro   = np.copy(self.S)
        if if_tanh:
            self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        else:
            self.y = (self.Jout@ self.S_ro + self.h_Jout)

        self.state_trace.append(self.S.copy())
        return self.H


    def reset (self):
        self.S   = np.zeros ((self.N,))
        self.S_hat   = np.zeros ((self.N,)) 
        self.S_ro   = np.zeros ((self.N,))
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  = np.zeros ((self.N,))
        self.y = np.zeros((self.O,))
        self.state_trace = []

    def display_reservoir_trace(self):
        """
        Display the reservoir state trace as a plot of N lines over time, every line in a different color
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        state_trace = np.array(self.state_trace)
        for neuron in range(self.N):
            if neuron==0:
                plt.plot(state_trace[:, neuron], label=f'Neuron {neuron+1}', alpha=0.5, color='red', linewidth=2, marker='s')
            elif neuron==len(self.state_trace[0]) - 1:
                plt.plot(state_trace[:, neuron], label=f'Neuron {neuron+1}', alpha=0.5, color='blue', linewidth=2, marker='s')
            else:
                plt.plot(state_trace[:, neuron], alpha=0.5)
        plt.title('Reservoir State Trace')
        plt.xlabel('Time Steps')
        plt.ylabel('State Value')
        plt.legend()
        plt.show()

    def train(self,X, Y, reg=0.01, tangent=True):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (n_samples, N+1)

        # Solve (ridge regression): W_out_aug = (X_aug^T X_aug + reg I)^(-1) X_aug^T Y
        X_aug_T = X_aug.T
        if tangent:
            W_out_aug = np.linalg.pinv(X_aug_T @ X_aug + reg * np.eye(X_aug.shape[1])) @ X_aug_T @ np.arctanh(Y)
        else:
            W_out_aug = np.linalg.pinv(X_aug_T @ X_aug + reg * np.eye(X_aug.shape[1])) @ X_aug_T @ Y

        self.Jout = W_out_aug[:-1, :].T  # shape: (O, N)
        self.h_Jout = W_out_aug[-1, :]

def reservoir_episode(agent, env, reservoir, time_steps=30):
    env.reset_inner()
    reservoir.reset()
    done = False
    time = 0
    traj = []
    reservoir_states = []
    while not done and time < time_steps:
        time += 1
        traj.append(env.agent_position.copy())
        agent_position = env.encoded_position
        action, probs = agent.sample_action(agent_position.flatten())
        reward, done = env.step(action)

        r_encoded = env.encode(reward)

        input_modulation = .1 + reservoir.Jin_mult * reward
        input_modulation = input_modulation.flatten()

        reservoir_input = np.concatenate((agent_position, probs, r_encoded.flatten()))
        reservoir.step_rate(reservoir_input, input_modulation)
        reservoir_states.append(reservoir.S.copy())

    return reward, np.array(traj), reservoir_states

def test_reservoir():
    reservoir = Reservoir()
    print(reservoir.Jin_mult.shape)

    from environment import Environment
    from agent import LinearAgent

    env = Environment()
    agent = LinearAgent()
    reward = 0
    while reward != 1.5:
        reward, traj, reservoir_states = reservoir_episode(agent, env, reservoir, time_steps=30)
    
    reservoir.display_reservoir_trace()

    while reward != 0:
        reward, traj, reservoir_states = reservoir_episode(agent, env, reservoir, time_steps=30)
    reservoir.display_reservoir_trace()

def test_reservoir_on_failures():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from environment import Environment
    from agent import LinearAgent

    reservoir = Reservoir()
    env = Environment()
    agent = LinearAgent()

    success_traces = []
    fail_traces = []

    print("Collecting traces...")

    # Gather 2 success and 2 failure episodes
    while len(success_traces) < 2 or len(fail_traces) < 2:
        reward, traj, trace = reservoir_episode(agent, env, reservoir, time_steps=30)
        if reward == 1.5 and len(success_traces) < 2:
            success_traces.append(np.array(trace))
        elif reward == 0 and len(fail_traces) < 2:
            fail_traces.append(np.array(trace))

    success_traces = [np.array(t) for t in success_traces]
    fail_traces = [np.array(t) for t in fail_traces]

    for i, trace in enumerate(success_traces):
        print(f"Success Trace {i+1} Shape: {trace.shape}")
    for i, trace in enumerate(fail_traces):
        print(f"Failure Trace {i+1} Shape: {trace.shape}")
    
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    all_traces = success_traces + fail_traces
    labels = ['Success 1', 'Success 2', 'Failure 1', 'Failure 2']

    for idx, trace in enumerate(all_traces):
        ax = axes[idx]
        trace = np.array(trace)  # shape: (T, N)
        T, N = trace.shape
        neurons = np.linspace(0, N-1, 8, dtype=int)

        for n in neurons:
            ax.plot(trace[:, n], alpha=0.8, label=f'N{n}')
        ax.set_title(labels[idx])
        ax.set_ylabel("State")
        ax.set_xlabel("Time step")
        if idx == 0:
            ax.legend(ncol=4, fontsize='small')

    plt.tight_layout()
    plt.suptitle("Reservoir Neuron Traces for Success and Failure Episodes", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    test_reservoir_on_failures()
    print("Reservoir initialized successfully.")