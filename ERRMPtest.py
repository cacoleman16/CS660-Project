
import math
import random
import argparse
import time
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd
import torchvision.transforms as T
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



class agent:
	def __init__(self):
		self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
		self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
		self.size=self.env.observation_space.shape
		self.options=self.env.action_space.n
		self.baseline=0
	def get_screen(self):
		self.env.render()
	def close(self):
		self.env.close()
	def doStep(self, a):
		sP, r, done, info = self.env.step(a)
		return r, done, sP




#Used some code from:
#https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
class Network(nn.Module):
	def __init__(self, sze, opt):
		super(Network, self).__init__()
		#Used some code from:
		#https://pytorch.org/docs/stable/generated/torch.nn.Module.html
		self.size=sze[0]*sze[1]*sze[2]
		self.dtype = torch.float
		self.options=opt
		self.H = 128
		self.N=256
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
		else:
			self.device = torch.device("cpu")

		self.model = nn.Sequential(
			nn.Linear(self.size, self.H), #, requires_grad=True
			nn.ReLU(),
			nn.Linear(self.H, self.N),
			nn.ReLU(),
			nn.Linear(self.N, self.H),
			nn.ReLU(),
			nn.Linear(self.H, self.options),
			nn.ReLU(),
			nn.Softmax(dim=-1)
			)

		self.model.to(self.device)
		for param in self.parameters():
			#print(param.grad.shape)
			param.retain_grad()
		self.requires_grad = True

		self.dtype = torch.float
		self.optimizer=optim.Adam(self.parameters())
		self.to(self.device)
		self.baseline=0
		self.avgRetn=np.zeros([1,300])


	def forward(self, s):
		state = torch.tensor(s, device=self.device, dtype=torch.float, requires_grad=True)
		distribution = self.model(state)
		a = np.random.choice(range(self.options), p=distribution.detach().numpy())
		return a, distribution[a]


	def reinforce(self, pi, v, s, a, alpha):
		#Get the return
		returns = torch.zeros(len(v), device=self.device, dtype=self.dtype, requires_grad=True)
		with torch.no_grad():
			returns[len(pi)-1]=v[len(v)-1]
			for ii in range(len(pi)-1):
				i=len(pi)-ii-1
				returns[i-1]=v[i-1]+returns[i]
		
		#Add onto the averages
		if len(self.avgRetn) > 1 or self.avgRetn[0][0] != 0:
			self.avgRetn=np.append(self.avgRetn,[[0]*300], axis=0)
		i=0
		while i < len(self.avgRetn[0]) and i < len(returns):
			self.avgRetn[len(self.avgRetn)-1][i]=returns[i]
			i+=1
		means = np.mean(self.avgRetn, axis=0)

		#loss
		loss = torch.zeros(len(v), device=self.device, dtype=self.dtype, requires_grad=True)
		with torch.no_grad():
			for i in range(len(pi)):
				if means[i] > 0:
					loss[i]= -1*math.log(pi[i])*(means[i]-returns[i])
				else:
					loss[i]= -1*math.log(pi[i])*returns[i]
		#print("LOSS: ",loss)


		for i in range(len(pi)):
			#print("")
			#self.forward(s[i])	#For printing only
			self.optimizer.zero_grad()
			loss[i].backward()
			#print(loss[i].backward())
			self.optimizer.step()
			#self.forward(s[i])	#For printing only
			#print(loss[i])
		
class controller: 
	def __init__(self, agnt):
		random.seed(time.localtime())
		self.gamma=.9
		self.agent = agnt
		self.training=True
		self.net = Network(self.agent.size,self.agent.options)

	def sumReward(self,rlist):
		total=0
		for reward in rlist:
			total += reward
		return(total)

	def act(self, runs):
		self.net.baseline=self.agent.baseline
		for i in range(runs):
			self.agent.get_screen()
			s=self.agent.env.reset()
			rewards = []
			if self.training:
				states=[]
				states.append(s)
				actionProbs = []
				actions = []
			done=False
			while done ==False:
				a, aProb = self.net.forward(s)
				#add aProb and state to set
				if self.training:
					actions.append(a)
					actionProbs.append(aProb.item())
					states.append(s)
				
				reward, done, sP= self.agent.doStep(a)
				rewards.append(reward.item())
				self.agent.get_screen()
				s=sP

			if self.training:
				self.net.reinforce(actionProbs,rewards,states, actions,0.0001)
			print(self.sumReward(rewards), self.net.baseline-self.sumReward(rewards))
		#print(self.net.parameters)
		#print(list(self.net.parameters())[0].grad)
		self.end()

	def end(self):
		self.agent.get_screen()
		self.agent.close()


parser = argparse.ArgumentParser(description='Define frozen lake environment and settings.')
parser.add_argument('--env', choices=['cart', 'lunar'], default='cart', help='The environment tested, can be cart or lunar')
parser.add_argument('--num_episodes', type=int, default = 5, help='The number of episodes run. In collection mode, this is the size of increments collected.')
parser.add_argument('--verbose', help='Print more information.', action='store_true')
args = parser.parse_args()

mario = agent()

use = controller(mario)
use.act(args.num_episodes)
