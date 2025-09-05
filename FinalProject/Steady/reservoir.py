import numpy as np

"""
The provided code implements a reservoir computing network, a specialized type of recurrent neural network architecture where only the output 
weights are trained while the internal recurrent connections remain fixed. This implementation uses a continuous-time 
rate-based model with heterogeneous time constants and stochastic dynamics.

Reservoir computing leverages a randomly connected, fixed recurrent network to transform input signals into high-dimensional representations. 
These representations capture complex temporal dependencies in the input data, making previously inseparable patterns linearly separable. 
Only the output weights (readout layer) are trained, typically using simple regression methods.

The network operates in a dynamical regime near the "edge of chaos" where the recurrent connections
are scaled to maintain rich dynamics without becoming unstable.

"""

class Reservoir:
    """
    Recurrent network of stochastic rate units with heterogeneous time constants.
    
    This implements a continuous-time reservoir computing model with the following features:
    - Fixed random recurrent connections scaled to a specified spectral radius
    - Heterogeneous neuron time constants (from fast to slow)
    - Stochastic state updates for robust dynamics
    - Configurable input and output connectivity
    """

    def __init__(self, par):
        # network sizes
        self.N, self.I, self.O, self.T = par['shape']

        # dynamics params
        self.dt = par['dt']
        self.tau_m = np.linspace(par['tau_m_f'], par['tau_m_s'], self.N)

        self.dv = par['dv']  # kept in case you reintroduce a smooth nonlinearity

        # recurrent / IO weights
        self.J = np.random.normal(0., par['sigma_rec'], size=(self.N, self.N))
        sr = 0.95
        eig_max = np.abs(np.linalg.eigvals(self.J)).max()
        self.J *= sr / eig_max

        self.Jin   = np.random.normal(0., par['sigma_input'],  size=(self.N, self.I))
        self.Jout  = np.random.normal(0., par['sigma_output'], size=(self.O, self.N))
        self.h_Jout = np.zeros((self.O,))

        # output buffer
        self.y = np.zeros((self.O,))

        self.name = 'model'

        # external field (kept: plausible external hook)
        h = par['h']
        assert type(h) in (np.ndarray, float, int)
        self.h = h if isinstance(h, np.ndarray) else np.ones(self.N) * h

        # membrane potential & state
        self.H  = np.ones(self.N,) * par['Vo'] * 0.0  # starts at 0.0
        self.Vo = par['Vo']  # kept in case external code inspects it

        self.S     = np.zeros(self.N,)
        self.S_hat = np.zeros(self.N,)
        self.S_ro  = np.zeros(self.N,)

        # store params
        self.par = par

    def step_rate(self, inp, inp_modulation, sigma_S, if_tanh=True):
        """
        Perform one time step of the reservoir dynamics.
        
        Implements a rate-based update with tanh nonlinearity in the membrane potential
        and Gaussian noise in the neuronal state.
        
        The membrane update follows the discretized dynamics:
        τᵢ·dHᵢ/dt = -Hᵢ + tanh(∑ⱼ Jᵢⱼ·Sⱼ + ∑ₖ Jinᵢₖ·inputₖ)
        # preserve previous spikes
        self.S_hat = np.copy(self.S)
        Args:
            inp (np.ndarray): Input vector at the current time step.
            inp_modulation (float): Scaling factor for the input.
            sigma_S (float): Standard deviation of the Gaussian noise added to the state.
            if_tanh (bool): Whether to apply tanh nonlinearity to the output readout.
        Returns:
            np.ndarray: Updated membrane potentials of the neurons.
        """
        # membrane update
        decay = np.exp(-self.dt / self.tau_m)
        self.H = (
            self.H * decay
            + (1 - decay) * np.tanh(self.J @ self.S_hat + (self.Jin @ inp) * inp_modulation)
        )

        # noisy state (rate surrogate)
        self.S = np.copy(self.H + np.random.normal(0., sigma_S, size=np.shape(self.H)))
        self.S_ro = np.copy(self.S)

        # readout
        lin = self.Jout @ self.S_ro + self.h_Jout
        self.y = np.tanh(lin) if if_tanh else lin

        return self.H

    def reset(self):
        """
        Reset all internal state variables to zeros.
        
        This resets the network to its initial state, clearing any temporal memory.
        """
        self.S      = np.zeros((self.N,))
        self.S_hat  = np.zeros((self.N,))
        self.S_ro   = np.zeros((self.N,))
        self.H      = np.zeros((self.N,))
        self.y      = np.zeros((self.O,))

def initialize_reservoir(neurons = 600):
    """
    Create and initialize a reservoir with standard parameters.
    
    This is a convenience function that creates a reservoir with typical parameters
    suitable for spatial-temporal tasks, the params are many, and the balance of the reservoir is easily perturbed. 
    This is the reason why we provide a standard initialization, if one wants to change parameters, they should do so in a controlled manner.
    Args:
        neurons (int): Number of neurons in the reservoir.
    """
    spatial_res = 5
    gradient_spectral_rad = 0.7
    dt = 0.001
    tau_m_f = 100.0 * dt
    tau_m_s = 1.0 * dt
    tau_s = 2.0 * dt
    tau_ro = 0.001 * dt
    beta_ro = np.exp(-dt / tau_ro)
    dv = 5.0
    alpha = 0.0
    Vo = 0.0
    h = 0.0
    s_inh = 0.0
    sigma_input = 0.08
    sigma_teach = 0.0
    sigma_output = 0.1
    offT = 1
    alpha_rout = 0.0001

    N = neurons
    input_dim = 4 + 10 + spatial_res**2 + 10
    output_dim = 4 * spatial_res**2
    TIME = 600
    shape = (N, input_dim, output_dim, TIME)

    par = {
        'tau_m_f': tau_m_f,
        'tau_m_s': tau_m_s,
        'tau_s': tau_s,
        'tau_ro': tau_ro,
        'beta_ro': beta_ro,
        'dv': dv,
        'alpha': alpha,
        'Vo': Vo,
        'h': h,
        's_inh': s_inh,
        'N': N,
        'T': TIME,
        'dt': dt,
        'offT': offT,
        'alpha_rout': alpha_rout,
        'sigma_input': sigma_input,
        'sigma_teach': sigma_teach,
        'sigma_rec': gradient_spectral_rad / np.sqrt(N),
        'shape': shape,
        'sigma_output': sigma_output
    }

    network_reservoire_gradient = Reservoir(par)
    network_reservoire_gradient.Jin_mult = np.random.normal(0, 0.1, size=(N,))
    return network_reservoire_gradient

def build_W_out(x_grad_net_coll, y_grad_coll, noise=5e-5):
    """
    Compute optimal output weights (W_out) using ridge regression.
    
    Solves for the linear mapping from reservoir states to target outputs
    with small noise regularization to improve generalization.
    Args:
        x_grad_net_coll (list of np.ndarray): Collected reservoir states over time.
        y_grad_coll (list of float): Corresponding target outputs.
        noise (float): Regularization noise level.
    Returns:
        np.ndarray: Optimal output weight vector.
    """
    X = np.array(x_grad_net_coll)
    y_grad_coll = np.clip(y_grad_coll, -0.999, 0.999)
    Y = np.arctanh(np.array(y_grad_coll)) 
    X_noisy = X + np.random.normal(0, noise, size=X.shape)

    return np.linalg.pinv(X_noisy) @ Y