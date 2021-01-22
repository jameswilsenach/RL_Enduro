import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

plotting = False
from enduro.agent import Agent
from enduro.action import Action
from enduro.state import EnvironmentState
episodes = 800

class QAgent(Agent):
    def __init__(self):
        super(QAgent, self).__init__()
        # The horizon defines how far the agent can see
        self.horizon_row = 5

        self.grid_cols = 10
        # The state is defined as a tuple of the agent's x position and the
        # x position of the closest opponent which is lower than the horizon,
        # if any is present. There are four actions and so the Q(s, a) table
        # has size of 10 * (10 + 1) * 4 = 440.
        self.Q = np.ones((self.grid_cols, self.grid_cols + 1, 4))
        self.episodes = episodes
        self.ep_rewards = np.zeros(self.episodes)
        self.Q_conv = np.zeros(self.episodes-1)
        self.current_Q = 0.
        # Add initial bias toward moving forward. This is not necessary,
        # however it speeds up learning significantly, since the game does
        # not provide negative reward if no cars have been passed by.
        self.Q[:, :, 0] += 1.

        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}

        # Learning rate
        self.alpha = 1e-4
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01

        # Log the obtained reward during learning
        self.last_episode = 1

    def initialise(self, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.
        """
        self.total_reward = 0

        self.next_state = self.buildState(grid)

    def converge_rate(self,current,last):
#       return np.log(np.abs((val_list[0]-val_list[1])/(val_list[1]-val_list[2])))/(np.log(np.abs((val_list[1]-val_list[2])/(val_list[2]-val_list[3]))))
        return np.sum(np.abs(current-last))

    def act(self):
        """ Implements the decision making process for selecting
        an action. Remember to store the obtained reward.
        """

        self.state = self.next_state
        self.choose_random = np.random.uniform(0., 1.) < self.epsilon
        # If exploring
        if self.choose_random:
            # Select a random action using softmax
            idx = np.random.choice(4)
            self.action = self.idx2act[idx]
        else:
            # Select the greedy action
            self.action = self.idx2act[self.argmaxQsa(self.state)]

        self.reward = self.move(self.action)
        self.total_reward += self.reward

    def sense(self, grid):
        self.next_state = self.buildState(grid)

        # Visualise the environment grid
        #cv2.imshow("Environment Grid", EnvironmentState.draw(grid))

    def learn(self):
        # Read the current state-action value
        Q_sa = self.Q[self.state[0], self.state[1], self.act2idx[self.action]]
        # Calculate the updated state action value
        Q_sa_new = Q_sa + self.alpha * (self.reward + self.gamma * self.maxQsa(self.next_state) - Q_sa)
        # Write the updated value
        self.Q[self.state[0], self.state[1], self.act2idx[self.action]] = Q_sa_new

    def callback(self, learn, episode, iteration):
        print("{0}/{1}: {2}".format(episode, iteration, self.total_reward))
        if self.choose_random == False:
            print("{0}/{1}: {2} Action: {3} (Exploiting!)".format(episode, iteration, self.total_reward,Action.toString(self.action)))
        else:
            print("{0}/{1}: {2} Action: {3} (Exploring!)".format(episode, iteration, self.total_reward,Action.toString(self.action)))
        if iteration >= 6500:
            self.ep_rewards[episode-1]=self.total_reward
            self.last_Q=self.current_Q+0.
            self.current_Q=self.Q.flatten()
            if episode>1:
                self.Q_conv[episode-2]=self.converge_rate(self.current_Q,self.last_Q)
                print(self.Q_conv[episode-2])
            if episode == self.episodes :
                print("Rewards Mean: {0:.2f}, Reward Variance: {1:.2f}".format(np.mean(self.ep_rewards),np.var(self.ep_rewards)))
                if plotting:    
                    fig1 = plt.figure()
                    plt.plot(np.arange(self.episodes)+1,self.ep_rewards)
                    plt.xlabel('episode')
                    plt.ylabel('reward')
                    plt.show()
                    fig1.savefig('rewards.pdf',format='pdf')

                    fig2 = plt.figure()
                    plt.bar(np.arange(len(self.Q)),self.Q,width=1)
                    plt.xlabel('index')
                    plt.xticks(np.arange(len(self.Q)))
                    plt.ylabel('weight')
                    plt.show()
                    fig2.savefig('qs.pdf',format='pdf')

                    fig3 = plt.figure()
                    plt.plot(np.arange(2,len(self.Q_conv)+2),self.Q_conv)
                    plt.xlabel('episode')
                    plt.ylabel('mean rate')
                    plt.show()
                    fig3.savefig('Q_conv.pdf',format='pdf')

                np.save('rewards',self.ep_rewards)
                np.save('Qs',self.current_Q)
                np.save('Q_conv',self.Q_conv)
        # Log the reward at the current iteration

    def buildState(self, grid):
        state = [0, 0]

        # Agent position (assumes the agent is always on row 0)
        [[x]] = np.argwhere(grid[0, :] == 2)
        state[0] = x

        # Sum the rows of the grid
        rows = np.sum(grid, axis=1)
        # Ignore the agent
        rows[0] -= 2
        # Get the closest row where an opponent is present
        rows = np.sort(np.argwhere(rows > 0).flatten())

        # If any opponent is present
        if rows.size > 0:
            # Add the x position of the first opponent on the closest row
            row = rows[0]
            for i, g in enumerate(grid[row, :]):
                if g == 1:
                    # 0 means that no agent is present and so
                    # the index is offset by 1
                    state[1] = i + 1
                    break
        return state

    def maxQsa(self, state):
        return np.max(self.Q[state[0], state[1], :])

    def argmaxQsa(self, state):
        return np.argmax(self.Q[state[0], state[1], :])


if __name__ == "__main__":
    a = QAgent()
    a.run(True, episodes=episodes, draw=False)
