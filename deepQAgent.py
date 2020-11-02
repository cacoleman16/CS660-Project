import gym
import random
import numpy as np
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class deepQAgent():

    def __init__(self, env):
        self.env = env
        self.esp = 0.99
        self.gamma = 0.99999
        self.alpha = 0.0003
        self.name = 'Deep  Q-learning'
        self.memorySize = 5000
        self.count = 0
        self.batchSize = 64
        # self.img_h, self.img_w, self.img_c = env.observation_space.shape
        self.input_shape = self.env.observation_space.shape
        # self.input_dims = self.memorySize * self.img_c
        self.intializeMemory()
        self.Q = QNetwork(
            alpha=self.alpha, n_actions=env.action_space.n,
            input_shape=self.input_shape)
        self.Q = self.Q.to(self.Q.device)

    def intializeMemory(self):
        self.stateMemory = torch.zeros((
            self.memorySize, * self.input_shape))
        self.nextStateMemory = torch.zeros((
            self.memorySize, * self.input_shape))
        self.rewardMemory = torch.zeros((
            self.memorySize, 1))
        self.terminalMemory = torch.zeros((self.memorySize, 1))
        self.actionMemory = torch.zeros((
            self.memorySize, 1))

    def getMax(self, currentState):
        # picks best action so far
        state = currentState.to(self.Q.device)
        exploit = torch.argmax(self.Q(state)).unsqueeze(0).unsqueeze(0).cpu()

        return exploit

    def selectionPolicy(self, currentState):
        p = np.random.random()

        if p < self.esp:
            # picks random action
            explore = self.env.action_space.sample()
            return explore
        else:
            exploit = self.getMax(currentState)
            return exploit

    def sampleReplay(self):
        # so I don 't pick from ones I haven't seen
        maxMemory = min(self.count, self.memorySize)
        batch = np.random.choice(maxMemory, self.batchSize, replace=False)

        batchIndex = np.arange(self.batchSize, dtype=np.int32)
        # select the batch from each tensor
        currentStateReplay = self.stateMemory[batch]
        nextStateReplay = self.nextStateMemory[batch]
        rewardReplay = self.rewardMemory[batch]
        terminalReplay = self.terminalMemory[batch]

        # int because torch is picky about dtypes
        actionReplay = self.actionMemory[batch]

        return currentStateReplay, nextStateReplay, rewardReplay, terminalReplay, actionReplay

    def Update(self, currentState, nextState, reward, action, done):
        index = self.count % self.memorySize
        self.stateMemory[index] = currentState
        self.nextStateMemory[index] = nextState
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = done
        self.count += 1

    def learn(self):
        if self.count < self.batchSize:
            return
        self.Q.optimzier.zero_grad()
        # sample a batch from the replay memoryory

        currentStateReplay, nextStateReplay, rewardReplay, terminalReplay, actionReplay = self.sampleReplay()

        currentStateReplay = currentStateReplay.to(self.Q.device)
        nextStateReplay = nextStateReplay.to(self.Q.device)
        rewardReplay = rewardReplay.to(self.Q.device)
        terminalReplay = terminalReplay.to(self.Q.device)
        actionReplay = actionReplay.to(self.Q.device)

        # needed when using a whole batch and not just one
        q = self.Q(currentStateReplay).gather(1, actionReplay.long())

        nextQ = self.Q(nextStateReplay).max(1).values.unsqueeze(1)

        # target = rewardReplay + self.gamma * torch.max(nextQ, dim=1)[0]

        target = rewardReplay + \
            torch.mul((self.gamma * nextQ), 1 - terminalReplay)

        # get the  squared intertemporal loss
        loss = self.Q.loss(target, q).to(self.Q.device)
        loss.backward()
        self.Q.optimzier.step()

    def train(self, env, esp_decline_num=.000004):
        # Reset will reset the environment to its initial configuration and return that state.
        currentState = env.reset()
        currentState = torch.Tensor([currentState])
        done = False
        stepCount = 0
        # Loop until either the agent finishes or takes 200 actions, whichever comes first.
        while stepCount < 20000 and done == False:
            stepCount += 1

            # pick an action using esp greedy
            actionToTake = self.selectionPolicy(currentState)
            action = int(actionToTake)
            # print("Action Taken: " + str(actionToTake))
            # Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
            nextState, reward, done, _ = env.step(action)

            nextState = torch.tensor([nextState])
            reward = torch.tensor([reward]).unsqueeze(0)
            done = torch.tensor([int(done)]).unsqueeze(0)

            self.Update(currentState, nextState, reward, actionToTake, done)

            # learn

            self.learn()
            # Render visualizes the environment
            # env.render()
            # make next State this state
            currentState = nextState
            if self.esp > esp_decline_num + sys.float_info.epsilon:
                self.esp -= esp_decline_num

    def simulation(self, agent, env, numEps=5000, testEps=20, divisor=50, esp_decline_num=.000004):
        total_rewards = np.ones(round(numEps/divisor))
        testCounter = 0
        for i in range(numEps):
            agent.train(env, esp_decline_num=esp_decline_num)
            print(i)
            if i % divisor == 0:
                cumulativeReward = 0
                # then go off policy
                for i in range(testEps):
                    # Reset will reset the environment to its initial configuration and return that state.
                    currentState = env.reset()
                    currentState = torch.Tensor([currentState])
                    done = False
                    stepCount = 0
                    # Loop until either the agent finishes or takes 200 actions, whichever comes first.
                    while stepCount < 200 and done == False:
                        stepCount += 1
                        # pick an action based on pi
                        actionToTake = agent.getMax(currentState)
                        action = int(actionToTake)
                        # print("Action Taken: " + str(actionToTake))
                        # Execute actions using the step function. Returns the nextState, reward, a boolean indicating whether this is a terminal state. The final thing it returns is a probability associated with the underlying transition distribution, but we shouldn't need that for this assignment.
                        nextState, reward, done, _ = env.step(action)
                        # intialize values for next state if doesnt exist yet

                        nextState = torch.tensor([nextState])
                        reward = torch.tensor([reward]).unsqueeze(0)
                        done = torch.tensor([int(done)]).unsqueeze(0)
                        cumulativeReward += reward
                        currentState = nextState

                total_rewards[testCounter] = cumulativeReward / testEps
                testCounter += 1
        env.close()
        return total_rewards, agent.name


class QNetwork(nn.Module):
    def __init__(self, alpha,  n_actions, input_shape):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.optimzier = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros((1, *shape), device=self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
