import numpy as np

def encode(pos, res = None):
    if res is None: res = 10

    pos = np.array (pos)
    if len (pos.shape) == 1: pos = pos.reshape (-1, 1)
    shape = pos.shape

    x =  np.clip(pos.T, -1,1).T

    mu_x = np.linspace (-1.,1., num = res).T
    s_x= np.diff ((-1.,1.), axis = 0).T / (res)

    enc_x = np.exp (-0.5 * ((x.reshape (-1, 1) - mu_x) / s_x)**2).T

    return np.array (enc_x)


def encode_state(pos, res = 5):

    pos = np.array (pos)
    if len (pos.shape) == 1: pos = pos.reshape (-1, 1)
    
    x =  np.clip(pos[0].T, -1,1).T

    mu_x = np.linspace (-1.,1., num = res).T
    s_x= np.diff ((-1.,1.), axis = 0).T / (res)
    enc_x = np.exp (-0.5 * ((x.reshape (-1, 1) - mu_x) / s_x)**2).T

    y =  np.clip(pos[1].T, -1,1).T
    mu_y = np.linspace (-1.,1., num = res).T
    s_y= np.diff ((-1.,1.), axis = 0).T / (res)

    enc_y = np.exp (-0.5 * ((y.reshape (-1, 1) - mu_y) / s_y)**2).T

    return np.array ([enc_x,enc_y])

class PlaceCellEncoder2D:
    def __init__(self, grid_size, sigma):
        """
        Initialize the place cell encoder.

        Args:
            grid_size (int): Number of grid points along each dimension.
            sigma (float): Standard deviation parameter for the Gaussian activation function.
        """
        self.grid_size = grid_size
        self.sigma = sigma
        self.grid_centers = self.generate_grid_centers()

    def generate_grid_centers(self):
        """
        Generate grid centers within the plane (-1, 1) x (-1, 1).

        Returns:
            numpy array: Array containing the centers of the place cell grids.
        """
        x = np.linspace(-.55, .55, self.grid_size)
        y = np.linspace(-.55, .5, self.grid_size)
        grid_centers = np.array([(x_coord, y_coord) for x_coord in x for y_coord in y])
        return grid_centers

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