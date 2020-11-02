import random
import numpy as np
from random import sample
import sys
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


class dynaQAgent():

    def __init__(self, env):
        self.Q = {}
        self.visited = {}
        self.env = env
        self.esp = 0.99
        self.gamma = 0.99999
        self.iterations = 0
        self.alpha = 0.1
        self.intializeQ()
        self.intializeTransition()
        self.name = 'dynaQ'
        self.TerminalState = []

    def getMax(self, state):
        v = list(self.Q[state].values())
        k = list(self.Q[state].keys())
        maxAction = k[v.index(max(v))]
        if max(v) != 0:
            return maxAction
        else:
            maxes = [i for i, x in enumerate(v) if x == max(v)]
            return k[sample(maxes, 1)[0]]

    def intializeQ(self):
        for state in range(self.env.observation_space.shape):
            self.Q[state] = {}
            for action in range(self.env.action_space.n):
                self.Q[state][action] = 0

    def intializeTransition(self):
        self.Transition = np.full((self.env.observation_space.n,
                                   self.env.action_space.n, self.env.observation_space.n), 0.000000001)
        self.R = np.zeros(self.env.observation_space.n)

    def selectionPolicy(self, currentState):
        p = np.random.random()

        if p < self.esp:
            # picks random action
            explore = self.env.action_space.sample()
            return explore
        else:
            # picks best action so far
            exploit = self.getMax(currentState)
            return exploit

    def Update(self, currentState, nextState, reward, action, done=False):
        if not done:
            self.Q[currentState][action] = self.Q[currentState][action] + self.alpha * (reward + self.gamma *
                                                                                        (self.Q[nextState][self.getMax(nextState)]) - self.Q[currentState][action])

        else:
            self.Q[currentState][action] = self.Q[currentState][action] + \
                self.alpha * (reward - self.Q[currentState][action])

    def model(self, state, action):
        mle = self.Transition[state, action, :] / \
            np.sum(self.Transition[state, action])

       # sample from the mle distrubtion
        draw = random.random()
        prob = 0
        for idx, statePrime in enumerate(mle):
            prob += statePrime
            if draw <= prob:
                newState = idx
                rewardHat = self.R[newState]
                return rewardHat, newState

    def planning(self, iterations):
        for n in range(iterations):
            state = sample(list(self.visited.keys()), 1)[0]
            action = sample(self.visited[state], 1)[0]
            rewardHat, newState = self.model(state, action)

            if newState in self.TerminalState:
                self.Update(state, newState, rewardHat, action, done=True)
            else:
                self.Update(state, newState, rewardHat, action)

    def train(self, env, esp_decline_num=.00004):
        # Reset will reset the environment to its initial configuration and return that state.
        currentState = env.reset()
        done = False
        stepCount = 0
        # Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while stepCount < 5000 and done == False:
            stepCount += 1
            # pick an action using esp greedy
            actionToTake = self.selectionPolicy(currentState)

            # check if taken before
            if currentState not in self.visited:
                self.visited[currentState] = []
            if actionToTake not in self.visited[currentState]:
                self.visited[currentState].append(actionToTake)

            # print("Action Taken: " + str(actionToTake))
            # Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state.
            nextState, reward, done, _ = env.step(actionToTake)
            if done:
                self.TerminalState.append(nextState)
            # intialize values for next state if doesnt exist yet
            self.Update(currentState, nextState, reward, actionToTake, done)

            self.Transition[currentState, actionToTake, nextState] += 1
            self.R[nextState] = reward

            self.planning(iterations=self.iterations)  # self.iterations)

            # Render visualizes the environment
            # env.render()
            currentState = nextState
            if self.esp > esp_decline_num + sys.float_info.epsilon:
                self.esp -= esp_decline_num
