# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:10:18 2018

@author: Shinelon
"""
import matlab.engine #for matlab's lqr
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
import numpy as np

LOAD_PRE_MODEL = True #if you wanna repretrain the AC model make it False
AC_MAX_ITERS = 5000
LR = 0.01
gamma = 0.9
Q = np.array([[0.5, 0, 0, 0],[0, 4, 0, 0],[0, 0, 2500, 0],[0, 0, 0, 0.01]])
#Q = np.array([[2, 0, 0, 0],[0, 4, 0, 0],[0, 0, 2000, 0],[0, 0, 0, 0.04]])
R = np.array([[1, 0],[0, 1]])
INIT_Q = matlab.single(Q.tolist()) 
FOR_R = matlab.single(R.tolist())

class ENV():
    def __init__(self):
        self.q1 = 0.5
        self.q2 = 5
        self.q3 = 2500
        self.q4 = 0.02
        self.Q = INIT_Q
        self.R = FOR_R
        self.ref_score = 0
        self.eng = matlab.engine.start_matlab()
        self.count = 0
        #self.agent = Agent()
        
    def step(self, a_q, reset=False):
        #minNum = min(a_q)
        #maxNum = max(a_q)
        #a_q = [0.4*(w-minNum)/(maxNum-minNum)+0.8 for w in a_q]
        #a_q = [0.4*w+0.8 for w in a_q]
        # renew the Q matrix:
        if not reset:
            self.count += 1
            q1 = a_q[0]*self.q1
            q2 = a_q[1]*self.q2 
            q3 = a_q[2]*self.q3
            q4 = a_q[3]*self.q4
            #q2 = self.q2 
            #q3 = self.q3
            #q4 = self.q4
            self.Q = matlab.single([[q1, 0, 0, 0], [0, q2, 0, 0], [0, 0, q3, 0], [0, 0, 0, q4]])
        # return value:
        count = 1
        total_r = 0.0
        next_state = [0.0]*8
        next_state[4:] = a_q
        over = False
        # cal
        agent = Agent()
        delta_s = agent.reset()
        Q = self.Q
        R = self.R
        for i in range(4):
            total_r -= abs(delta_s[i])
            #next_state[i] += (abs(delta_s[i]) - next_state[i])*1./count
            next_state[i] += ((delta_s[i]) - next_state[i])*1./count
        while True:
            delta_r, delta_vt, delta_vr, delta_m = delta_s
            r, vt, vr, m = agent.state[agent.now]
            r += delta_r
            vt += delta_vt
            vr += delta_vr
            m += delta_m
            Tt_ref, Tr_ref = agent.control[agent.now]
            A = [[0, 0, 1, 0],[-vt*vr/(r**2), vr/r, vt/r, Tt_ref/(m**2)],\
                 [vt**2/(r**2)-2*4.9e12/(r**3), -2*vt/r, 0, Tr_ref/(m**2)],[0, 0, 0, 0]]
            B = [[0, 0], [-1./m, 0], [0, -1./m], [-np.sign(Tt_ref)/9.8/2940, -np.sign(Tr_ref)/9.8/2940]]
            A = matlab.single(A)
            B = matlab.single(B)
            #Q = self.Q
            #R = self.R
            K = self.eng.lqr(A, B, Q, R)
            delta_t = np.dot(-np.array(K), delta_s)
            delta_s, done = agent.step(delta_t)
            for i in range(4):
                count += 1
                total_r -= abs(delta_s[i])
                #next_state[i] += (abs(delta_s[i]) - next_state[i])*1./count
                next_state[i] += ((delta_s[i]) - next_state[i])*1./count
            if done == True:
                break
        if self.count >= AC_MAX_ITERS:
            over = True
        return next_state, total_r/count , over
    
    def reset(self):
        a_q = [self.q1, self.q2, self.q3, self.q4]
        s, self.ref_score, over = self.step(a_q, True)
        return s
    
    def render(self):
        pass
        

class Agent():
    def __init__(self):
        self.now = None
        self.nowState = None
        self.new_control_list = []
        self.new_state = []
        self.costate = []
        self.delta_control = []
        self.delta_state = []
        self.control = []
        self.time = []
        self.state = []
        self.state_dim = 8
        self.action_dim = 2
        self.action_bound = [7500, 7500]
        self.traindata_for_actor = []
        self.traindata_for_critic = []
        self.make()
    
    def make(self):
        with open('D:/matlab_new_control_list.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.new_control_list.append(line)
        with open('D:/matlab_new_state.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.new_state.append(line)
        with open('D:/matlab_State.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.state.append(line)
        with open('D:/matlab_Control.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.control.append(line)
        with open('D:/matlab_Time.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.time.append(line)
        with open('D:/matlab_costate.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.costate.append(line)
        with open('D:/matlab_delta_control_list.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.delta_control.append(line)
        with open('D:/matlab_delta_state_list.txt', 'r') as f:
            for line in f:
                line = line.strip().split()
                line = [float(w) for w in line]
                self.delta_state.append(line)   
        assert len(self.time) == len(self.new_state) == len(self.state) == len(self.control) == len(self.new_control_list)
        for i in range(len(self.time)-1):
            temp = []
            temp.extend(self.delta_state[i])
            temp.extend(self.delta_control[i])
            self.traindata_for_actor.append(temp)
        for i in range(len(self.time)-1):
            temp = []
            temp.extend(self.delta_state[i])
            temp.extend(self.costate[i])
            self.traindata_for_critic.append(temp)
        
    def reset(self):
        self.now = 0
        self.nowState = []
        self.nowState.extend((np.array(self.new_state[0])-np.array(self.state[0])).tolist())
        return self.nowState
    
    def step(self, action):
        self.now += 1
        Tt = action[0] + self.control[self.now-1][0]
        Tr = action[1] + self.control[self.now-1][1]
        stateRef = self.state[self.now-1]
        r = self.nowState[0] + stateRef[0]
        vt = self.nowState[1] + stateRef[1]
        vr = self.nowState[2] + stateRef[2]
        m = self.nowState[3] + stateRef[3]
        T = self.time[self.now][0] - self.time[self.now-1][0]
        next_r = r - vr*T
        next_vt = vt + (vt*vr/r - Tt/m)*T
        next_vr = vr + (-(Tr/m) - vt*vt/r + 4.9e12/r/r)*T
        next_m = m + (-(abs(Tt)+abs(Tr))/2940.0/9.80665)*T
        nextState = [next_r, next_vt, next_vr, next_m]
        self.nowState = (np.array(nextState)-np.array(self.state[self.now])).tolist()
        done = False
        if self.now == len(self.control)-1:
            done = True
        assert isinstance(self.nowState, list), len(self.nowState) == 4
        return self.nowState, done
    
    def render(self):
        pass
    
    def getprebuffer(self):
        return
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1_1 = nn.Linear(8, 160) # Q4 + E4 = 8
        self.fc2_1 = nn.Linear(160, 80)
        self.fc3_1 = nn.Linear(80, 1)
        
        self.fc1_2 = nn.Linear(8, 160) # Q4 + E4 = 8
        self.fc2_2 = nn.Linear(160, 80)
        self.fc3_2 = nn.Linear(80, 1)
        
        self.fc1_3 = nn.Linear(8, 160) # Q4 + E4 = 8
        self.fc2_3 = nn.Linear(160, 80)
        self.fc3_3 = nn.Linear(80, 1)
        
        self.fc1_4 = nn.Linear(8, 160) # Q4 + E4 = 8
        self.fc2_4 = nn.Linear(160, 80)
        self.fc3_4 = nn.Linear(80, 1)
        
    def forward(self, x):
        x1 = F.sigmoid(self.fc1_1(x))
        x1 = F.sigmoid(self.fc2_1(x1))
        x1 = self.fc3_1(x1)
        
        x2 = F.sigmoid(self.fc1_2(x))
        x2 = F.sigmoid(self.fc2_2(x2))
        x2 = self.fc3_2(x2)
        
        x3 = F.sigmoid(self.fc1_3(x))
        x3 = F.sigmoid(self.fc2_3(x3))
        x3 = self.fc3_3(x3)
        
        x4 = F.sigmoid(self.fc1_4(x))
        x4 = F.sigmoid(self.fc2_4(x4))
        x4 = self.fc3_4(x4)
        
        x = t.cat((x1, x2, x3, x4))
        x = F.sigmoid(x)
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(12, 308) # Q4 + E4 + nQ4 = 12
        self.fc2 = nn.Linear(308, 100)
        self.fc3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

max_score = -pow(2,30)
final_result = []
  
criterion = nn.MSELoss()  
actor = Actor()
actor_optimizer = optim.Adadelta(actor.parameters())

critic = Critic()
critic_optimizer = optim.Adadelta(critic.parameters()) 

env = ENV()
s = env.reset()
s = V(t.Tensor(s), requires_grad=True)
while True:
    a = actor(s)

    critic_input1 = t.cat((s,a))

    critic_output = critic(critic_input1)
    #actor_optimizer.zero_grad()
    #actor_loss = criterion(critic_output, V(t.Tensor([-7]), requires_grad=False))
    #actor_loss.backward(retain_graph=True)
    #actor_optimizer.step()
    ns, r, done = env.step(a.data.numpy().tolist())
    if r > max_score:
        max_score = r
        temp = a.data.numpy().tolist()
        final_result = env.Q
    print (r, a.data.numpy().tolist())
    if done == True:
        break
    ns = V(t.Tensor(ns), requires_grad=True)
    na = actor(ns)
    list_s = ns.data.numpy().tolist()
    list_a = na.data.numpy().tolist()
    list_s.extend(list_a)
    
    critic_input2 = V(t.Tensor(list_s), volatile=True)
    critic_target = r + gamma*critic(critic_input2)
    critic_loss = criterion(critic_output, critic_target)
    critic_optimizer.zero_grad()
    actor_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    actor_optimizer.step()
    s = ns









