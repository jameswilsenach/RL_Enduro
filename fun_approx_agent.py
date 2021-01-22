import cv2
import numpy as np
import matplotlib.pyplot as plt
plotting = False
from enduro.agent import Agent
from enduro.action import Action
episodes = 800

"""
My Functional Approximation-based Q-learning Agent for the Enduro Atari Game. Part of the Reinforcement Learning Course of my MSc in Informatics at The University of Edinburgh
"""
class FunctionApproximationAgent(Agent):
    def __init__(self,depth=6):
        super(FunctionApproximationAgent, self).__init__()
        # Helper dictionaries that allow us to move from actions to
        # Q table indices and vice versa
        self.idx2act = {i: a for i, a in enumerate(self.getActionsSet())}
        self.act2idx = {a: i for i, a in enumerate(self.getActionsSet())}
        self.episodes = episodes
        self.ep_rewards = np.zeros(self.episodes)
        # Learning rate
        self.alpha = 1e-4
        # Discounting factor
        self.gamma = 0.9
        # Exploration rate
        self.epsilon = 0.01
        # Sensing depth of agent
        self.depth = depth

        self.init = False
        self.episode = 0
        self.theta_conv = np.zeros(self.episodes-1)
        self.current_thetas = 0.

    def centroid(self,road):
        corners = [road[11][0],road[11][-1]]
        centre = np.mean(corners,axis=0)
        return centre
    
    # function to detect displacement of agent from road centre using logit normalisation
    def displacement(self,cars,road):
        centre = self.centroid(road)
        carloc = cars['self'][:2]
        return 1/(1+np.exp((carloc[0]-centre[0])/100))
    
    #function to build feature space
    def buildFeatures(self, road, cars, speed, grid):
        #action specific bias features
        a_bias = np.identity(4)
        features = a_bias
        
        # displacement from centre of road features
        dist = self.displacement(cars,road)
        not_dist = np.identity(4)[[1,2],:]*(1-dist)
        dist = np.identity(4)[[1,2],:]*dist
        features = np.concatenate((features,dist,not_dist),axis=0)

        [[x]] = np.argwhere(grid[0, :] == 2)
        carpos = x
        
        #detects if/if not cars exist directly in front
        infront = np.any(grid[:self.depth,carpos]==1)
        not_infront = np.identity(4)[[0,3],:]*(not infront)
        infront = np.identity(4)[[0,3],:]*infront
        features = np.concatenate((features,infront,not_infront),axis=0)
        
        #detects if/if not cars exist directly in front in column to right
        if carpos<9:
            inright = np.any(grid[:self.depth,carpos+1]==1)
        else:
            inright = 0
        not_inright = np.identity(4)[[1,2],:]*(not inright)
        inright = np.identity(4)[[1,2],:]*inright
        features = np.concatenate((features,inright,not_inright),axis=0)
        
        #detects if/if not cars exist directly in front in column to left
        if carpos>0:
            inleft = np.any(grid[:self.depth,carpos-1]==1)
        else:
            inleft = 0
        not_inleft = np.identity(4)[[1,2],:]*(not inleft)
        inleft = np.identity(4)[[1,2],:]*inleft
        features = np.concatenate((features,inleft,not_inleft),axis=0)
        
        #variable to modulate speed by acceleration/breaking
        speed0 = speed>0
        not_speed0 = np.identity(4)[[0,3],:]*(not speed0)
        speed0 = np.identity(4)[[0,3],:]*speed0
        features = np.concatenate((features,speed0,not_speed0),axis=0)
        
        #detects and modulates recovery from collisions
        collider = self.collision(cars)
        not_collider = np.identity(4)[[0,3],:]*(not collider)
        collider = np.identity(4)[[0,3],:]*collider
        features = np.concatenate((features,collider,not_collider),axis=0)

        # traffic = len(cars['others'])>0
        # traffic = np.identity(4)[[3],:]*traffic
        # features = np.concatenate((features,traffic),axis=0)
        print len(features)
        return features


    def evalQsa(self,features):
        Q = np.zeros(4)
        for i in range(4):
            Q[i]=np.dot(self.thetas,features[:,i])
        return Q

    def maxQsa(self, features):
        return np.max(self.evalQsa(features))

    def argmaxQsa(self, features):
        return np.argmax(self.evalQsa(features))

    def converge_rate(self,current,last):
        return np.sum(np.abs(current-last))

    def initialise(self, road, cars, speed, grid):
        """ Called at the beginning of an episode. Use it to construct
        the initial state.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            grid  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_features = self.buildFeatures(road, cars, speed, grid)
        if not self.init:
            self.thetas = np.ones(self.next_features.shape[0])/self.next_features.shape[0]
            #accelerate action bias
            self.thetas[0] += self.thetas[0]
            self.init = True
        # Reset the total reward for the episode
        self.total_reward = 0

    def act(self):
        """ Implements the decision making process for selecting
        an action.
        """
        self.features = self.next_features
        self.choose_random = np.random.choice(2,p=(1-self.epsilon,self.epsilon)) # Chooses whether to explore or exploit with probability 1-self.epsilon
        # Selects the best action index in current state
        if self.choose_random:
            self.chosenA = np.random.choice(4)
        else:
            self.chosenA = self.argmaxQsa(self.features)
        # Records reward for printing and performs action
        self.action = self.idx2act[self.chosenA]
        # Execute the action and get the received reward signal
        self.reward = self.move(self.action)
        self.total_reward += self.reward
        # IMPORTANT NOTE:
        # 'action' must be one of the values in the actions set,
        # i.e. Action.LEFT, Action.RIGHT, Action.ACCELERATE or Action.BRAKE
        # Do not use plain integers between 0 - 3 as it will not work

    def sense(self, road, cars, speed, grid):
        """ Constructs the next state from sensory signals.

        Args:
            road  -- 2-dimensional array containing [x, y] points
                     in pixel coordinates of the road grid
            cars  -- dictionary which contains the location and the size
                     of the agent and the opponents in pixel coordinates
            speed -- the relative speed of the agent with respect the others
            gird  -- 2-dimensional numpy array containing the latest grid
                     representation of the environment

        For more information on the arguments have a look at the README.md
        """
        self.next_features = self.buildFeatures(road, cars, speed, grid)

    def learn(self):
        """ Performs the learning procedure. It is called after act() and
        sense() so you have access to the latest tuple (s, s', a, r).
        """
        Qsa = self.evalQsa(self.features)[self.chosenA]
        print Qsa
        dQ =  self.alpha*(self.reward + self.gamma * self.maxQsa(self.next_features) - Qsa)
        self.thetas += dQ*self.features[:,self.chosenA]
        print self.thetas
        # self.thetas /= np.sqrt(np.sum(np.power(self.thetas,2)))

    def callback(self, learn, episode, iteration):
        self.episode = episode
        """ Called at the end of each timestep for reporting/debugging purposes.
        """
        if self.choose_random == False:
            print "{0}/{1}: {2} Action: {3} (Exploiting!)".format(episode, iteration, self.total_reward,Action.toString(self.action))
        else:
            print "{0}/{1}: {2} Action: {3} (Exploring!)".format(episode, iteration, self.total_reward,Action.toString(self.action))
        if iteration >= 6500:
            self.ep_rewards[episode-1]=self.total_reward
            self.last_thetas = self.current_thetas+0.
            self.current_thetas = self.thetas+0.
            print 'Here'+str(np.sum(self.current_thetas))+'  ' + str(np.sum(self.last_thetas))
            if episode>1:
                print 'Here'
                self.theta_conv[episode-2]=self.converge_rate(self.current_thetas,self.last_thetas)
                print self.theta_conv[episode-2]
            print self.thetas
            if episode == self.episodes :
                print "Rewards Mean: {0:.2f}, Reward Variance: {1:.2f}".format(np.mean(self.ep_rewards),np.var(self.ep_rewards))
                if plotting:
                    fig1 = plt.figure()
                    plt.plot(np.arange(self.episodes)+1,self.ep_rewards)
                    plt.xlabel('episode')
                    plt.ylabel('reward')
                    plt.show()
                    fig1.savefig('rewards.pdf',format='pdf')

                    fig2 = plt.figure()
                    plt.bar(np.arange(len(self.thetas)),self.thetas,width=1)
                    plt.xlabel('index')
                    plt.xticks(np.arange(len(self.thetas)))
                    plt.ylabel('weight')
                    plt.show()
                    fig2.savefig('thetas.pdf',format='pdf')

                    fig3 = plt.figure()
                    plt.plot(np.arange(4,len(self.theta_conv)+4),self.theta_conv)
                    plt.xlabel('episode')
                    plt.ylabel('rate')
                    plt.show()
                    fig3.savefig('theta_conv.pdf',format='pdf')

                np.save('rewards',self.ep_rewards)
                np.save('thetas',self.thetas)
                np.save('theta_conv',self.theta_conv)


if __name__ == "__main__":
    a = FunctionApproximationAgent()
    a.run(True, episodes=episodes, draw=True)
    print 'Total reward: ' + str(a.total_reward)
