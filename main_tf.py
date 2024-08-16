import gym 
from simple_tf import DeepQNetwork, Agent 
from utils import plotLearning 
from gym import wrappers 
import numpy as np 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.00005 
    n_games = 500 
    agent = Agent(gamma= 0.99, epsilon = 1.0, alpha = lr, input_dims = [8],
                  n_actions = 4, mem_size = 10000, n_games = n_games,
                  batch_size = 64)
    
    alpha = 'alpha' + str(lr)
    filename = '0-lunar-lander-256x256-' + alpha + 'base-adam-faster_decay.png'
    scores = []
    eps_history = []
    score = 0 
    env = wrappers.Monitor(env, "Simple-DL-TF/lunar-lander-4",
                           video_callable = lambda episode_id : True, force = True)
                
    
    for i in range(n_games):
        done = False 
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-10) : (i+1)])
            print("episode" , i, 'score', score, 
                   'average score %.3f' % avg_score, 
                   'epsilon %.3f' % agent.epsilon)
        else:
            print('episode' , i , 'score' , score)

        
        observation = env.reset() 
        score = 0 
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward 
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn() 
        eps_history.append(agent.epsilon)
        scores.append(score)
    
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)