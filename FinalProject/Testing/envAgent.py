import numpy as np
import matplotlib.pyplot as plt

spatial_res = 5  
sigma_distribution = 0

class Environment:
    def __init__(self):
        self.agent_position = np.array([0, 0])  # Agent starts at the center
        self.food_position = self._generate_food_position(0,0)
        self.square_size = 1.0

    def _generate_food_position(self,x,y,theta0=45):
        #return np.random.uniform(-1, 1, size=(2,))
        sx,sy=[sigma_distribution,sigma_distribution]
        #return np.random.normal([x,y], [sx,sy], size=(2,))
        theta = np.random.uniform(theta0 - sigma_distribution, theta0 + sigma_distribution)/180*np.pi
        return .5*np.cos(theta),.5*np.sin(theta)
    
    def reset(self,x,y, theta0=45):
        self.agent_position = np.array([0, 0])  # Reset agent position
        self.food_position = np.round(self._generate_food_position(x,y,theta0), decimals=1)  # Generate new food position

    def reset_inner(self):
        self.agent_position = np.array([0, 0])  # Reset agent position
        #self.food_position = self._generate_food_position()  # Generate new food position

    def step(self, action):
        # Define possible actions: 0 for moving left, 1 for moving right, 2 for moving up, 3 for moving down
        if action == 0:
            self.agent_position[0] -= 0.1
        elif action == 1:
            self.agent_position[0] += 0.1
        elif action == 2:
            self.agent_position[1] += 0.1
        elif action == 3:
            self.agent_position[1] -= 0.1

        # Clip agent position to be within the square
        self.agent_position = np.clip(self.agent_position, -self.square_size / 2, self.square_size / 2)

        # Calculate reward
        distance_to_food = np.linalg.norm(self.agent_position - self.food_position)
        reward = 0  # Reward is handled by the agent
        done = False
  
        #Check if the agent reached the food

        if distance_to_food < 0.15:
            reward += 10  # Reward for reaching the food
            
        if distance_to_food < 0.075:
            reward += 5  # Additional reward for getting close to the food
            done = True
            
        return reward, done

    def render(self, lims = 0.75):
        plt.figure(figsize=(5, 5))
        plt.xlim(-lims, lims)
        plt.ylim(-lims, lims)
        plt.plot(self.agent_position[0], self.agent_position[1], 'bo', markersize=10, label='Agent')
        plt.plot(self.food_position[0], self.food_position[1], 'rx', markersize=10, label='Food')
        plt.legend()
        plt.title('Agent Environment')
        plt.grid(True)
        plt.show()

        
class LinearAgent:
    def __init__(self, learning_rate=2.):
        self.weights = np.random.rand(4, spatial_res**2)  # Weights for the linear policy
        self.learning_rate = learning_rate

    def policy(self, state):
        logits = np.dot(self.weights, state)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        return probs.reshape(-1)  # Ensure action_probs is one-dimensional
