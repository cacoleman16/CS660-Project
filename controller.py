from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from dynaQAgent import dynaQAgent
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from envWrapper import make_env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# turns the frame into a smaller greyscale and sets joypad to simple
env = make_env(env)
# env = JoypadSpace(env, SIMPLE_MOVEMENT)


done = True
for step in range(5000):
    if done:
        state = env.reset()
    action = agent.getMax(state)
    state, reward, done, info = env.step(action)
    env.render()

env.close()
