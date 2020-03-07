import numpy as np
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        state_space = 4
        action_space = 2
        # print(state_space,action_space)
        num_hidden = 128

        self.l1 = nn.Linear(state_space, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, action_space, bias=False)

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.5),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x).to(device)


class Tester(object):

    def __init__(self):
        """
        Initialize the Tester object by loading your model.
        """
        # TODO: Load your pyTorch model for Policy Gradient here.
        # values = np.zeros(16)

        self.model = Policy()
        self.model.load_state_dict(torch.load("policy_model.pt"))
        self.model.eval()


    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
        """Evaluate the value of a policy.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.
        
        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        # TODO: Your Code Goes Here
        
        iterations = max_iterations
        numStates = env.nS
        numActions = env.nA
        

        # values = self.values
        values = np.zeros(16)
        old_values = np.zeros(16)
        delta = 1000

        while(iterations>0 and delta>tol):
          iterations -= 1  
          
          for state in range(numStates):
            action = policy[state]
            p,newstate,reward,isterminal = np.array(env.P[state][action]).T
            newstate = np.array(newstate,dtype=np.int32)
            isterminal = np.array(isterminal,dtype=np.bool)
            
            values[state] = np.dot(p,(reward + gamma*values[newstate]))

          delta = np.amax(np.absolute(values-old_values))
          # print(values,delta)
          old_values = np.copy(values)  

        num_evaluations = max_iterations-iterations  
        return values, num_evaluations

    def policy_gradient_test(self, state):
        """
        Parameters
        ----------
        state: np.ndarray
            The state from the CartPole gym environment.
        Returns
        ------
        np.ndarray
            The action in this state according to the trained policy.
        """
        # TODO. Your Code goes here.
        # print(state)
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action_probs = self.model(state).cpu()
        distributions = Categorical(action_probs)
        action = distributions.sample().numpy()
        return action