import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, grid_size = 5, sigma = 0.2, spatial_res = None, sigma_distribution = None):

        self.spatial_res = spatial_res if spatial_res is not None else 5
        self.sigma_distribution = sigma_distribution if sigma_distribution is not None else 10
        self.agent_position = np.array([0, 0])  
        self.food_position = self._generate_food_position(0.0)
        self.square_size = 1.0
        self.grid_size = grid_size
        self.sigma = sigma
        self.grid_centers = self._generate_grid_centers()
        self.encoded_position = self.encode_position(self.agent_position)

    def _generate_grid_centers(self):
        """
        Generate grid centers within the plane (-1, 1) x (-1, 1).

        Returns:
            numpy array: Array containing the centers of the place cell grids.
        """
        x = np.linspace(-.55, .55, self.grid_size)
        y = np.linspace(-.55, .5, self.grid_size)
        grid_centers = np.array([(x_coord, y_coord) for x_coord in x for y_coord in y])
        return grid_centers

    def _generate_food_position(self, theta0=45):
        # theta = np.random.uniform(theta0 - self.sigma_distribution, theta0 + self.sigma_distribution)/180*np.pi
        theta = theta0 / 180 * np.pi
        return .5*np.cos(theta),.5*np.sin(theta)

    def reset(self, theta0=45):
        self.agent_position = np.array([0, 0])  # Reset agent position
        self.encoded_position = self.encode_position(self.agent_position)
        self.food_position = np.round(self._generate_food_position(theta0), decimals=1)  # Generate new food position

    def reset_inner(self):
        self.agent_position = np.array([0, 0])  # Reset agent position
        self.encoded_position = self.encode_position(self.agent_position)

    def encode_position(self, position):
        """
        Encode a 2D position into place cell activations.

        Args:
            position (tuple): A tuple containing the x and y coordinates of the position.

        Returns:
            numpy array: Activation levels of place cells.
        """
        x, y = position
        activation_levels = np.exp(-((x - self.grid_centers[:, 0])**2 + (y - self.grid_centers[:, 1])**2) / (2 * self.sigma**2))
        return activation_levels

    def encode(self, x, res=None):
        if res is None: res = 10
        x = np.array(x).T
        mu_x = np.linspace(-1.,1., num = res).T
        s_x= np.diff((-1.,1.), axis = 0).T / (res)
        enc_x = np.exp (-0.5 * ((x.reshape (-1, 1) - mu_x) / s_x)**2).T
        return enc_x

    def step(self, action):
        # Calculate new position without clipping
        new_position = self.agent_position.copy()
        if action == 0:
            new_position[0] -= 0.1
        elif action == 1:
            new_position[0] += 0.1
        elif action == 2:
            new_position[1] += 0.1
        elif action == 3:
            new_position[1] -= 0.1

        # Calculate reward BEFORE clipping
        distance_to_food = np.linalg.norm(new_position - self.food_position)
        reward = 0
        done = False
        
        if distance_to_food < 0.15:
            reward += 1
        if distance_to_food < 0.075:
            reward += 0.5
            done = True

        # Now clip and update position
        self.agent_position = np.clip(new_position, -self.square_size/2, self.square_size/2)
        self.encoded_position = self.encode_position(self.agent_position)
        
        return reward, done

    def render(self, ax=None, title='Agent Environment', lims=0.75):
        """
        Render the current state of the environment into the given Axes.

        Args:
            ax (matplotlib.axes.Axes, optional): If provided, draw into this Axes.
            lims (float): Limits for the plot axes.

        Returns:
            matplotlib.axes.Axes: The Axes object containing the plot.
        """
        # if no Axes was passed in, create a new figure+axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        ax.plot(self.agent_position[0],
                self.agent_position[1],
                'bo',
                markersize=10,
                label='Agent')
        ax.plot(self.food_position[0],
                self.food_position[1],
                'rx',
                markersize=10,
                label='Food')
        ax.legend()
        ax.set_title(title)
        ax.grid(True)
        return ax


if __name__ == "__main__":
    env = Environment(grid_size=10, sigma=0.2)
    env.reset()
    print("Food Position:", env.food_position)
    print("Agent Position:", env.agent_position)

    encoded_position = env.encode_position((0.1, 0.2))
    print("Encoded Position:", encoded_position)
    print("Encoded Position Shape:", encoded_position.shape)
    print("Reward Encoded:", env.encode(0, res=5))

    reward, done = env.step(1)
    print("Reward:", reward, "Done:", done)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.grid(True)
    ax.set_title('Agent Environment - All Resets')
    ax.plot(env.agent_position[0], env.agent_position[1], 'bo', markersize=10, label='Agent (start)')

    n_resets = 8
    n_angle = 0

    for n in range(n_resets):
        theta0 = 45 * n_angle
        n_angle += 1
        env.reset(theta0=theta0)
        ax.plot(env.food_position[0], env.food_position[1], 'rx', markersize=8)

    ax.legend(["Agent (start)", "Food positions"])
    plt.show()