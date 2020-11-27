import gym
import random
import numpy as np
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import sys
import gc


class ActorCriticAgent():

    def __init__(self, env, actor_weights=None, critic_weights=None):
        # self.stepLimit = 20000
        self.env = env
        self.n_inputs = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.actorNet = ActorNet(
            n_inputs=self.n_inputs, n_actions=self.n_actions)
        self.criticNet = CriticNet(
            n_inputs=self.n_inputs, n_actions=self.n_actions)
        self.gamma = 0.95
        self.score = 0
        self.xpos = None
        self.broken = False
        self.initializeReinforce()
        if actor_weights != None:
            self.actorNet.load_state_dict(torch.load(
                actor_weights, map_location=self.actorNet.device))
        if critic_weights != None:
            self.criticNet.load_state_dict(torch.load(
                critic_weights, map_location=self.criticNet.device))

    def initializeReinforce(self):
        self.log_probabilities = []
        self.q_values = []
        self.rewards = []

    def getAction(self, currentState):

        currentState = currentState.to(self.actorNet.device)
        logits = self.actorNet(currentState).unsqueeze(0).unsqueeze(0).cpu()
        try:
            distrubtion = Categorical(F.softmax(logits, dim=1))
            actionToTake = distrubtion.sample()

            if self.log_probabilities == []:
                self.log_probabilities = distrubtion.log_prob(actionToTake)

            else:
                self.log_probabilities = torch.cat(
                    [self.log_probabilities, distrubtion.log_prob(actionToTake)])
            actionToTake = actionToTake.item()

        except:
            print("Didn't work so tried random")
            self.broken = True
            actionToTake = self.env.action_space.sample()

        return actionToTake

    def onehotEncodeAction(self, action, shape):
        onehot = torch.zeros((shape), dtype=torch.long)
        index = action % self.n_actions
        onehot[0, index] = 1
        return onehot

    def discountRewards(self):
        discounted_rewards = []
        # normalize discounted rewards
        for t in range(len(self.rewards)):
            reward_to_go = 0
            power = 0
            for r in self.rewards[t:]:
                reward_to_go += (self.gamma**power) * r
                power += 1
            discounted_rewards.append(reward_to_go)

        return discounted_rewards

    def normalizeRewards(self, rewards):

        rewards = torch.tensor(
            rewards)
        normalized_rewards = (rewards - rewards.mean()) \
            / (rewards.std() + sys.float_info.epsilon)

        return normalized_rewards

    def policyUpdate(self):

      # zero gradients
        self.actorNet.optimizer.zero_grad()
        self.criticNet.optimizer.zero_grad()

        rewards = self.discountRewards()
        rewards = self.normalizeRewards(rewards).detach()

        q_values = torch.cat(self.q_values, 0)
        advantage_function = rewards - q_values
        self.log_probabilities = self.log_probabilities
        weighted_negative_liklihoods = - \
            (self.log_probabilities * advantage_function.detach())

        # get loss
        advantage_function_loss = advantage_function.pow(
            2).mean().to(self.actorNet.device)
        gradient = weighted_negative_liklihoods.mean().to(self.actorNet.device)

        # auto diff
        gradient.backward()
        # retain_graph=True)
        advantage_function_loss.backward()

        # clip to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(self.criticNet.parameters(), 5)
        # print(gradient)
        torch.nn.utils.clip_grad_norm_(self.actorNet.parameters(), 5)

        self.criticNet.optimizer.step()
        self.actorNet.optimizer.step()

        del gradient, advantage_function_loss

    def train(self, env):
        step = 0
        batch_size = 128
        no_move = 0
        lives = 3
        doingPoorly = 0
        currentState = env.reset()
        currentState = torch.Tensor([currentState]).float()
        # .to(self.actorNet.device)
        done = False
        while done == False:
            step += 1

            # take action
            actionToTake = self.getAction(currentState)
            value = self.criticNet(currentState.to(self.actorNet.device))

            # store the action and states
            self.q_values.append(value.unsqueeze(0).unsqueeze(0).cpu())
            action = int(actionToTake)
            del actionToTake

            nextState, reward, done, info = env.step(action)

            reward = info['score'] - self.score
            self.score = info['score']

            if lives != info['life']:
                reward -= 300
                lives = info['life']

            if not self.xpos:
                self.xpos = info['x_pos']
                self.stage = info['stage']
            elif (self.xpos < info['x_pos']) & (self.stage == info['stage']):
                reward = reward - 500
                no_move += 1
                if (no_move == 20) and (reward < 0):
                    reward -= 1000
                    doingPoorly += 1
                if doingPoorly == 50:
                    reward -= 10000
                    done = True

            self.xpos = info['x_pos']
            self.stage = info['stage']

            # covert to tensor
            nextState = torch.tensor([nextState]).float()

            self.rewards.append(reward)

            currentState = nextState

            if step % batch_size == 0:
                if self.q_values != []:
                    self.policyUpdate()
                    self.initializeReinforce()

        if self.q_values != []:
            self.policyUpdate()
            self.initializeReinforce()

    def test(self, env,  render=False):
        step = 0
        cumlativeReward = 0
        done = False
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = make_env(env)
        currentState = env.reset()
        currentState = torch.Tensor(
            [currentState]).float().to(self.actorNet.device)
        while done == False:
            step += 1
            actionToTake = self.getAction(currentState)
            action = int(actionToTake)
            nextState, reward, done, _ = env.step(action)
            cumlativeReward += reward
            nextState = torch.tensor([nextState]).float().to(
                self.actorNet.device)
            currentState = nextState

        # env.close()
        return cumlativeReward

    def simulation(self, env, numEps=5000, testEps=20, divisor=50):
        # need to reward the rewards and agent name
        cumlativeReward = np.ones(int(numEps / divisor))
        counter = 0
        for episode in range(numEps):
            self.train(env)
            print(episode)
            if episode % divisor == 0:
                rewards = 0
                for i in range(testEps):
                    r = self.test(env, render=False)
                    rewards += r
                cumlativeReward[counter] = rewards / testEps
                counter += 1

        return cumlativeReward


class ActorNet(nn.Module):
    def __init__(self, n_inputs, n_actions, alpha=0.0001):
        super(ActorNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv_out_size = self._get_conv_out(n_inputs)

        self.seq_model = nn.Sequential(nn.Linear(self.conv_out_size, 64),
                                       nn.Dropout(p=0.5),
                                       nn.ReLU(),
                                       nn.Linear(64, n_actions)
                                       ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros((1, *shape), device=self.device))
        return int(np.prod(o.size()))

    def forward(self, state):

        conv_out = self.conv(state).view(state.size()[0], -1)

        return self.seq_model(conv_out)


class CriticNet(nn.Module):
    def __init__(self, n_inputs, n_actions, alpha=0.0001):
        super(CriticNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.conv_out_size = self._get_conv_out(n_inputs)

        self.seq_model2 = nn.Sequential(nn.Linear(self.conv_out_size, 64),
                                        nn.ReLU(),
                                        nn.ReLU(),
                                        nn.Linear(64, 1)
                                        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.loss = nn.CrossEntropyLoss(reduction='none')

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros((1, *shape), device=self.device))
        return int(np.prod(o.size()))

    def forward(self, state):

        conv_out = self.conv(state).view(state.size()[0], -1)

        return self.seq_model2(conv_out)
