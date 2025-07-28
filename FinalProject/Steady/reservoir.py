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
                 ridge_lambda=0e-6,
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

        self.tanh = False

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

    def predict(self, state, tanh=True):
        """
        state shape (n_res,)
        returns gradient vector of shape (n_out,)
        """
        if self.tanh:
            return np.tanh(self.W_out @ state + self.bias)
        else:
            return (self.W_out @ state + self.bias) 

    def reset_state(self):
        """Set the internal activation vector x to zero."""
        self.state[:] = 0.  

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