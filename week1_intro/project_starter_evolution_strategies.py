
# coding: utf-8

# ### Project :: Evolution Strategies
# 
# ![img](https://t4.ftcdn.net/jpg/00/17/46/81/240_F_17468143_wY3hsHyfNYoMdG9BlC56HI4JA7pNu63h.jpg)
# 
# Remember the idea behind Evolution Strategies? Here's a neat [blog post](https://blog.openai.com/evolution-strategies/) about 'em.
# 
# Can you reproduce their success? You will have to implement evolutionary strategies and see how they work.
# 
# This project is optional; has several milestones each worth a number of points [and swag].
# 
# __Milestones:__
# * [10pts] Basic prototype of evolutionary strategies that works in one thread on CartPole
# * [+5pts] Modify the code to make them work in parallel
# * [+5pts] if you can run ES distributedly on at least two PCs
# * [+10pts] Apply ES to play Atari Pong at least better than random
# * [++] Additional points for all kinds of cool stuff besides milestones
# 
# __Rules:__
# 
# * This is __not a mandatory assignment__, but it's a way to learn some cool things if you're getting bored with default assignments.
# * Once you decided to take on this project, please tell any of course staff members so that we can help ypu if you get stuck.
# * There's a default implementation of ES in this [openai repo](https://github.com/openai/evolution-strategies-starter). It's okay to look there if you get stuck or want to compare your solutions, but each copy-pasted chunk of code should be understood thoroughly. We'll test that with questions.

# ### Tips on implementation
# 
# * It would be very convenient later if you implemented a function that takes policy weights, generates a session and returns policy changes -- so that you could then run a bunch of them in parallel.
# 
# * The simplest way you can do multiprocessing is to use [joblib](https://www.google.com/search?client=ubuntu&channel=fs&q=joblib&ie=utf-8&oe=utf-8)
# 
# * For joblib, make sure random variables are independent in each job. Simply add `np.random.seed()` at the beginning of your "job" function.
# 
# Later once you got distributed, you may need a storage that gathers gradients from all workers. In such case we recommend [Redis](https://redis.io/) due to it's simplicity.
# 
# Here's a speed-optimized saver/loader to store numpy arrays in Redis as strings.
# 
# 

# ## Conclusion
# time without joblib between updates of w = 66.70272040367126
# 

# In[8]:


import gym

from time import time
from joblib import Parallel, delayed
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle


# In[15]:

env = gym.make('CartPole-v0')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# In[16]:


from torch.autograd import Variable
import torch.nn.functional as F


class SimpleDense(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, n_actions=2, obs_shape=4):
        super(SimpleDense, self).__init__()
        
        self.fc1 = torch.nn.Linear(obs_shape, 8)
        self.fc2 = torch.nn.Linear(8, n_actions)
        self.softmax = torch.nn.Softmax(1)
        
    def forward(self, x):
        x = torch.from_numpy(x.reshape(1, 4)).float()
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return torch.argmax(x).numpy()
    
    def get_weights(self):
        """conv, fc1, fc2"""
        return self.fc1.weight, self.fc2.weight
    
    def assign_weights(self, fc1, fc2):
        """input tensor"""
        self.fc1.weight = torch.nn.Parameter(fc1)
        self.fc2.weight = torch.nn.Parameter(fc2)


# In[19]:

model = SimpleDense()
#act = model.forward(env.reset())
#print(act)


# In[20]:


#model.conv1.weight = torch.nn.Parameter(model.conv1.weight + 0.01 * torch.rand_like(model.conv1.weight))


# In[21]:


def play_game(model, env, n=100):
    s = env.reset()
    reward_history = []
    for i in range(n):
        #env.render()
        s,r,done, _ = env.step(model.forward(s))
        reward_history.append(r)
        if done:
            #print('done')
            #break
            s = env.reset()
            reward_history.append(-10.0)
    return reward_history


# In[22]:


def init_model(model, weight):
    pass

def get_update(w1, w2, sigma, model, env, n_games=10):
    jit1 = sigma * torch.rand_like(w1)
    jit2 = sigma * torch.rand_like(w2)
    model.assign_weights(w1 + jit1, w2 + jit2)
    reward = np.sum(play_game(model, env, n_games))
    return jit1, jit2, reward
    


n_games = 100
n = 60
epoch = 10
sigma = 0.5
lern_rate = 0.2
hist_reward = []
save_path = "models.npz"

w1 = torch.rand_like(model.fc1.weight)
w2 = torch.rand_like(model.fc2.weight)
shape1 = list(w1.shape)
shape2 = list(w2.shape)
shape1.append(n)
shape2.append(n)
w1_stack = torch.zeros(shape1)
w2_stack = torch.zeros(shape2)
reward_stack = torch.zeros(n)



#models = []
#for i in range(n):
#    models.append(SimpleDense())
models = pickle.load(open(save_path, 'rb'))

t = time()    
for j in range(epoch):
    ##for i in range(n):
    ##    w1_stack[:,:,i], w2_stack[:,:,i], reward_stack[i] = get_update(w1, w2, sigma, models[-1], env, n_games)
    res = Parallel(n_jobs=2)(delayed(get_update)(w1, w2, sigma, models[i], env, n_games) for i in range(n))
    for i in range(n):
        w1_stack[:,:,i], w2_stack[:,:,i], reward_stack[i] = res[i]
    
    A = (reward_stack - torch.mean(reward_stack)) / (torch.std(reward_stack) + 1e-9)
    w1 += lern_rate * torch.matmul(w1_stack, A) 
    w2 += lern_rate * torch.matmul(w2_stack, A)
    hist_reward.append(sum(reward_stack))
    print('fin epoch ', j, t - time())

pickle.dump(models, open(save_path, 'wb'))

import matplotlib.pyplot as plt
plt.plot(hist_reward)
plt.show()


