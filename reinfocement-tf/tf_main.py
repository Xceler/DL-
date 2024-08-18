import numpy as np 
import gym 
import matplotlib.pyplot as plt 
from tf_reni import PolicyGradientAgent 
from utils import plotLearning 
from gym import wrappers 

if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA = 0.0005, input_dims = 8, GAMMA = 0.99,
                                n_actions = 4, layer1_size = 64, layer2_size = 64,
                                chkpt_dir = 'reinforcement-tf/lunar-lander-ckpt')

                    
    
    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0 
    num_episodes = 2500 
    for i in range(num_episodes):
        print('episode:', i, 'score: ', score)
        done = False 
        score = 0 
        observation = env.reset() 
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done , info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_ 
            score += reward 
        score_history.append(score)
        agent.learn() 
