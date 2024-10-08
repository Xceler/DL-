import matplotlib.pyplot as plt 
import numpy as np 
import gym 

def plotLearning(x, scores, epsilons, filename, lines = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label = '1')
    ax2 = fig.add_subplot(111, label = '2', frame_on = False)

    ax.plot(x, epsilons, color = 'C0')
    ax.set_xlabel('Game', color = 'C0')
    ax.set_ylabel('Epsilon' , color = 'C0')
    ax.tick_params(axis ='x', colors ='C0')
    ax.tick_params(axis = 'y', colors = 'C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
        ax2.scatter(x, running_avg, color = 'C1')
        ax2.axes.get_axis().set_visible(False)
        ax2.yaxis.tick_right() 
        ax2.set_ylabel('Score', color = 'C1')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis = 'y', colors = 'C1')

        if lines is not None:
            for line in lines:
                plt.axvline(x =line)
            
        
        plt.savefig(filename) 

class SkipEnv(gym.Wrapper):
    def __init__(self, env = None, skip =4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip 
    
    def step(self, action):
        t_reward = 0.0 
        done = False 
        for _ in range(self._skip):
            obs, reward, done ,info = self.env.step(action)
            t_reward += reward 
            if done:
                break 
        
        return obs, t_reward, done, info 
    
    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset() 
        self._obs.buffer.append(obs)
        return obs 
    


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env = None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low =0, high = 225 , shape =(80, 80, 1), dtype =np.unit8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)
    
    @staticmethod 
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299 * new_frame[:,:,0] + 0.587 * new_frame[:, : , 1]+ \
                    0.114 * new_frame[:,:,2]
                

        new_frame = new_frame[35:195:2, ::2].reshape(80,80, 1)
        return new_frame.astype(np.unit8)

    

class MoveImageChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImageChannel, self).__init__(env)
        self.observation_space = gym.space.Box(low = 0.0, high = 1.0,
                                 shape =(self.observation_space.shape[-1],
                                         self.observation_space.shape[0],
                                         self.observation_space.shape[1]),
                                dtype = np.float32)
        
    def observation(self, observation):
        return np.moveaxis(observation, 2,0)
    

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32)/ 255.0 
    

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                 env.observation_space.low.repeat(n_steps, axis = 0),
                                 env.observation_space.high.repeat(n_steps, axis=0),
                                 dtype = np.float32
        )
    
    def reset(self):
        sef.buffer= np.zeros_like(self.observation_space.low, dtype = np.float32)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation 
        return self.buffer 
    

def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImageChannel(env, 4)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)