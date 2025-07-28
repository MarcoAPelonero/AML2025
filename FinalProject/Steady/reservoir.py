import numpy as np

class ModulatedESN:
    """
    Echo–State Network with a single scalar modulator (e.g. reward).
    state_t+1 = (1-α)*state_t + α*tanh( W_res @ state_t
                                       + (1 + β*mod_t) * (W_in @ inp_t) + W_bias )
    The read‑out is trained by ridge regression.
    """
    def __init__(self,
                 n_in,                 # dimensionality of ordinary inputs  u_t
                 n_res=500,            # reservoir size
                 spectral_radius= 0.76,  # spectral radius of W_res
                 input_scaling=.1,
                 mod_scaling=0.2,       # β above
                 leak_rate=0.6,        # α above  (α=1 => no leaky integration)
                 ridge_lambda=1e-4,
                 seed=None):
        rs = np.random.RandomState(seed)
        self.n_in, self.n_res = n_in, n_res
        self.leak = leak_rate
        self.mod_scaling = mod_scaling
        self.ridge_lambda = ridge_lambda

        W = rs.randn(n_res, n_res)
        eigs = np.abs(np.linalg.eigvals(W))
        self.W_res = W * (spectral_radius / eigs.max())

        self.W_in = rs.randn(n_res, n_in) * input_scaling

        self.W_bias = rs.uniform(-.1, .1, size=(n_res,))

        self.W_out = None
        self.state = np.zeros(n_res)

        self.tanh = True

    def step(self, u_t, r_t):
        """Advance one step.  u_t: (n_in,),  r_t: scalar modulator."""
        pre = ((self.W_res @ self.state) + (self.W_in @ u_t) + self.W_bias)
        x_new = np.tanh((1 + self.mod_scaling * r_t) * pre)
        self.state = (1 - self.leak) * self.state + self.leak * x_new
        return self.state.copy()        
    
    def collect(self, inputs, mods, washout=50):
        """
        Run one episode and return reservoir states (after wash‑out) and inputs.
        inputs  shape (T, n_in)
        mods    shape (T,)   – reward or other modulator
        """
        states = []
        for u_t, r_t in zip(inputs, mods):
            self.step(u_t, r_t)
            states.append(self.state.copy())
        return np.array(states[washout:]), inputs[washout:]

    def fit(self, X_state, Y_target):
        """
        Ridge regression read‑out with intercept:
          W_raw = Y Xᵀ (X Xᵀ + λI)⁻¹

        X_state shape (N, n_res+1)  ← note the +1 intercept
        Y_target  shape (N, n_out)
        """
        noise_size = 0.0000005

        W_out = np.linalg.pinv(X_state + np.random.normal(0, noise_size, size=np.shape(X_state))).dot(np.arctanh(np.array(Y_target)))

        self.W_out = W_out.T
        self.bias = np.zeros(self.W_out.shape[0])  # Bias term for the output layer

    def predict(self, state=None):
        """
        state shape (n_res,)
        returns gradient vector of shape (n_out,)
        """
        if state is None:
            state = self.state
        if self.tanh:
            return np.tanh(self.W_out @ state + self.bias)
        else:
            return (self.W_out @ state + self.bias) 

    def reset_state(self):
        """Set the internal activation vector x to zero."""
        self.state[:] = 0.  


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

    def step(self, inp, inp_modulation,sigma_S=0.00, if_tanh=True):
        self.S_hat   = np.copy(self.S)
        self.H   = self.H   * np.exp(-self.dt/self.tau_m) + (1-np.exp(-self.dt/self.tau_m) ) * np.tanh( ( self.J @ self.S_hat   + self.Jin @ inp ) * (inp_modulation) )

        self.S  = np.copy(self.H + np.random.normal (0., sigma_S, size = np.shape(self.H) ))
        self.S_ro   = np.copy(self.S)
        self.state_trace.append(self.S.copy())
        return self.H


    def reset_state(self):
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

    def fit(self,X, Y, reg=0.0001, tangent=True):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (n_samples, N+1)

        # Solve (ridge regression): W_out_aug = (X_aug^T X_aug + reg I)^(-1) X_aug^T Y
        X_aug_T = X_aug.T
        if tangent:
            W_out_aug = np.linalg.pinv(X_aug_T @ X_aug + reg * np.eye(X_aug.shape[1])) @ X_aug_T @ np.arctanh(Y)
        else:
            W_out_aug = np.linalg.pinv(X_aug_T @ X_aug + reg * np.eye(X_aug.shape[1])) @ X_aug_T @ Y

        self.Jout = W_out_aug[:-1, :].T  # shape: (O, N)
        self.h_Jout = W_out_aug[-1, :]

    def predict(self):
        self.y = np.tanh(self.Jout@ self.S_ro + self.h_Jout)
        return self.y.copy()

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

def plot_reservoir_states(reservoir_states, title='Reservoir States'):
    # Plot the reservoir states as a plot of N neurons lines over time
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    for i in range(reservoir_states.shape[1]):
        plt.plot(reservoir_states[:, i], label=f'Neuron {i+1}')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Activation Level')
    plt.grid()
    plt.show()
    
def episode(agent, env, reservoir, time_steps=30, verbose=False):
    
    env.reset_inner()
    reservoir.reset_state()

    trajectories = []
    reservoir_states = []

    done = False
    t = 0
    while not done and t < time_steps:
        action, probs = agent.sample_action(env.encoded_position)
        reward, done = env.step(action)

        reservoir_input = np.concatenate((env.encoded_position, probs, env.encode(reward, res=4).flatten()))
        reservoir_state = reservoir.step(reservoir_input, reward)

        if verbose:
            print(f"Step {t + 1}/{time_steps}, Action: {action}, Reward: {reward}")

        trajectories.append(env.agent_position.copy())
        reservoir_states.append(reservoir_state.copy())
        t += 1

    max_length = time_steps
    padded_traj = np.full((max_length, env.agent_position.shape[0]), np.nan)  
    padded_traj[:len(trajectories), :] = trajectories
    
    padded_reservoir_states = np.full((max_length, reservoir_states[0].shape[0]), np.nan)
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states

    return reward, padded_traj, padded_reservoir_states

def train_episode(agent, env, reservoir, time_steps=30, verbose=False):
    
    env.reset_inner()
    reservoir.reset_state()

    trajectories = []
    reservoir_states = []

    done = False
    t = 0
    while not done and t < time_steps:
        action, probs = agent.sample_action(env.encoded_position)
        reward, done = env.step(action)

        reservoir_input = np.concatenate((env.encoded_position, probs, env.encode(reward, res=4).flatten()))
        reservoir_state = reservoir.step(reservoir_input, reward)

        if verbose:
            print(f"Step {t + 1}/{time_steps}, Action: {action}, Reward: {reward}")

        trajectories.append(env.agent_position.copy())
        reservoir_states.append(reservoir_state.copy())
        t += 1

    max_length = time_steps
    padded_traj = np.full((max_length, env.agent_position.shape[0]), np.nan)  
    padded_traj[:len(trajectories), :] = trajectories
    
    padded_reservoir_states = np.full((max_length, reservoir_states[0].shape[0]), np.nan)
    padded_reservoir_states[:len(reservoir_states), :] = reservoir_states

    return reward, padded_traj, padded_reservoir_states

def main():
    from agent import LinearAgent
    from environment import Environment

    env = Environment()
    agent = LinearAgent()

    print(env.encoded_position)

    _, probs = agent.sample_action(env.encoded_position)


    print("Action Probabilities:", probs)

    print("Encoded Position Shape:", env.encoded_position.shape)
    print("Probs Shape:", probs.shape)
    print("Encoded reward shape:", env.encode(0, res=4))

    inputs0 = np.concatenate((env.encoded_position, probs, env.encode(0, res=4).flatten())) 

    print("Inputs0 Shape:", inputs0.shape)
    print("Inputs0:", inputs0)

    res = ModulatedESN(n_in= 33, n_res=500, seed=42,)
    out = res.step(inputs0, 0)
    print("Reservoir Output Shape:", out.shape)
    print(out)

    reward = 0
    counter = 0
    while reward < 0.5:
        counter += 1
        reward, trajectories, reservoir_states = episode(agent, env, res, time_steps=30, verbose=False)

    # Un-pad reservoir states for plotting
    reservoir_states = reservoir_states[~np.isnan(reservoir_states).any(axis=1)]
    print("Length of Reservoir States:", len(reservoir_states))
    plot_reservoir_states(reservoir_states, title='Reservoir States Over Time')

    print(f"Episode {counter}, Reward: {reward}")

if __name__ == "__main__":
    main()