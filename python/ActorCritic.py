# -*- coding: utf-8 -*-
"""
Spyder编辑器

这是一个临时脚本文件
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np
from scipy import io as scio

LOAD_PRE_MODEL = True #if you wanna repretrain the AC model make it False
PRE_MAX_ITERS = 400000 #pretrain iters
AC_MAX_ITERS = 3
LR = 0.01
Q = np.array([[0.5, 0, 0, 0],[0, 4, 0, 0],[0, 0, 2500, 0],[0, 0, 0, 0.01]])
R = np.array([[1, 0],[0, 1]])


class ENV():
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
        #print(nextState, self.state[self.now])
        self.nowState = (np.array(nextState)-np.array(self.state[self.now])).tolist()
        #self.nowState.extend()
        #self.nowState.extend(self.state[self.now])
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
        self.fc1 = nn.Linear(4, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 2)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4, 308)
        self.fc2 = nn.Linear(308, 100)
        self.fc3 = nn.Linear(100, 4)
    
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

def get_train_set(name, batchsize):
    env = ENV()
    if name == "Actor":
        traindata = random.sample(env.traindata_for_actor, batchsize)
        trainX = []
        trainY = []
        for item in traindata:
            trainX.append(item[:4])
            trainY.append(item[4:])
    if name == "Critic":
        traindata = random.sample(env.traindata_for_critic, batchsize)
        trainX = []
        trainY = []
        for item in traindata:
            trainX.append(item[:4])
            trainY.append(item[4:])
    return t.Tensor(trainX), t.Tensor(trainY)


        
    
####### Initiate the value for Theta_l(Actor) and Yita_l(Critic) #########
""" Use pretrain method to initial the weights so that the convergence
    difficulty of the controller can be reduced when searching in lar-
    ge scale
"""
####################pretrain for actor#########################
def pre_train_for_actor(LOAD_PRE_MODEL = LOAD_PRE_MODEL):
    running_loss = 0
    actor = Actor()
    actor.cuda()
    if not LOAD_PRE_MODEL:
        for i in range(PRE_MAX_ITERS):
            #for data
            trainX, trainY = get_train_set("Actor", 32)
            inputs, target = Variable(trainX), Variable(trainY)
            inputs = inputs.cuda()
            target = target.cuda()
            #for optim
            criterion = nn.MSELoss()
            optimizer = optim.Adagrad(actor.parameters(), lr=LR)
            #zero grad
            optimizer.zero_grad()
            #forward and backword
            outputs = actor(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            #renew paremeters
            optimizer.step()
            #print loss
            running_loss += loss.data[0]
            if i%2000 == 1999:
                print ('[%5d] loss: %.3f' \
                       %(i+1, running_loss/2000))
                if running_loss/2000 < 40:
                    t.save(actor, 'C:/Users/Shinelon/Desktop/fangzhen/actor.pkl')
                    break
                running_loss = 0.0
    else:
        actor = t.load('C:/Users/Shinelon/Desktop/fangzhen/actor.pkl')
    return actor
        
###############pretrain for critic#########################
def pre_train_for_critic(LOAD_PRE_MODEL = LOAD_PRE_MODEL):
    running_loss = 0.0
    critic = Critic()
    critic.cuda()
    if not LOAD_PRE_MODEL:
        for i in range(PRE_MAX_ITERS):
            #for data
            trainX, trainY = get_train_set("Critic", 32)
            inputs, target = Variable(trainX), Variable(trainY)/1000
            inputs = inputs.cuda()
            target = target.cuda()
            #for optim
            criterion = nn.MSELoss()
            optimizer = optim.Adagrad(critic.parameters(), lr=LR)
            #zero grad
            optimizer.zero_grad()
            #forward and backword
            outputs = critic(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            #renew paremeters
            optimizer.step()
            #print loss
            running_loss += loss.data[0] 
            if i%2000 == 1999:
                print ('[%5d] loss: %.3f' \
                       %(i+1, running_loss/2000))
                if running_loss/2000 < 66:
                    t.save(critic, 'C:/Users/Shinelon/Desktop/fangzhen/critic.pkl')
                    break
                running_loss = 0.0
    else:
        critic = t.load('C:/Users/Shinelon/Desktop/fangzhen/critic.pkl')
    return critic      
####################### Body Part ###############################
"""
    Use the NAC to make the AC network converge so that we
    can get the final Actor network for tracking control
"""
def one_iteration(actor, critic):
    env = ENV()
    s = env.reset()
    s = Variable(t.Tensor(s), requires_grad=True).cuda()
    for i in range(10000):
        """
            renew theta for actor
        """
        # optim and zero_grad
        optimizer = optim.SGD(actor.parameters(), lr=0.001e-10, momentum=0.9)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        # forward + backword
        lamda = 1000*critic(s) #get costate
        Uk = actor(s) #input the state
        numpys = s.data.cpu().numpy()
        rk = np.dot(R, Uk.data.cpu().numpy()) + np.dot(np.array([[0, 0], \
                                         [-1./(numpys[3]+env.state[env.now][3]), 0], \
                                         [0, -1./(numpys[3]+env.state[env.now][3])], \
                                         [-np.sign(env.control[env.now][0])/2940.0/9.8, \
                                        -np.sign(env.control[env.now][1])/2940.0/9.8]]).T, lamda.data.cpu().numpy().T)
        loss = criterion(t.matmul(Variable(t.Tensor(rk).cuda(), requires_grad=False), Uk), Variable(t.Tensor([0]).cuda()))
        #loss = criterion(Uk, Variable(t.Tensor(Uk.data.cpu().numpy()/2).cuda(), requires_grad=False))
        loss.backward()
        optimizer.step()
    
        """
            step forward
        """
        Uk = actor(s) # use new Theta to cal Uk
        s, done = env.step(Uk.data.cpu().numpy())
        if done == True:
            break
        s = Variable(t.Tensor(s), requires_grad=True).cuda()
        """
            renew yita for critic
        """
        # optim and zero_grad
        optimizer = optim.SGD(critic.parameters(), lr=0.001e-10, momentum=0.9)
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        # forward + backward
        lamda = 1000*critic(s) #get costate
        numpys = s.data.cpu().numpy()
        r = numpys[0] + env.state[env.now][0]
        vt = numpys[1] + env.state[env.now][1]
        vr = numpys[2] + env.state[env.now][2]
        m = numpys[3] + env.state[env.now][3]
        Tt_ref = env.control[env.now][0]
        Tr_ref = env.control[env.now][1]
        rk = np.dot(R, Uk.data.cpu().numpy()) + np.dot(np.array([[0, 0], \
                                         [-1./(numpys[3]+env.state[env.now][3]), 0], \
                                         [0, -1./(numpys[3]+env.state[env.now][3])], \
                                         [-np.sign(env.control[env.now][0])/2940.0/9.8, -np.sign(env.control[env.now][1])/2940.0/9.8]]).T, lamda.data.cpu().numpy())
        #calculate the matrix M for deltaU/deltaX 
        M = np.zeros((4,2)) 
        trail = s + Variable(t.Tensor([0.1, 0, 0, 0])).cuda()
        U0 = actor(trail)
        M[0] = (U0-Uk).data.cpu().numpy()/0.1
        trail = s + Variable(t.Tensor([0, 0.1, 0, 0])).cuda()
        U1 = actor(trail)
        M[1] = (U1-Uk).data.cpu().numpy()/0.1
        trail = s + Variable(t.Tensor([0, 0, 0.1, 0])).cuda()
        U2 = actor(trail)
        M[2] = (U2-Uk).data.cpu().numpy()/0.1
        trail = s + Variable(t.Tensor([0, 0, 0, 0.1])).cuda()
        U3 = actor(trail)
        M[3] = (U3-Uk).data.cpu().numpy()/0.1
    
        lamda_target = np.dot(np.array([[0, 0, 1, 0], \
                                    [-vt*vt/r**2, vr/r, vt/r, Tt_ref/m**2], \
                                    [vt**2/r**2 - 2*4.9e12/r**3, -2*vt/r, 0, Tr_ref/m**2], \
                                    [0, 0, 0, 0]]), lamda.data.cpu().numpy()) + \
                    np.dot(Q, numpys) +\
                    np.dot(M, rk)
        loss = criterion(lamda, Variable(t.Tensor(lamda_target).cuda(), requires_grad=False))
        loss.backward()
        optimizer.step()
    return actor, critic

############################ Get result! ################################
def result_output(actor):
    result = []
    env = ENV()
    s = env.reset()
    s = Variable(t.Tensor(s).cuda())
    while True:
        a = actor(s)
        result.append(a.data.cpu().numpy().tolist())
        s, done = env.step(a.data.cpu().numpy())
        if done:
            break
        s = Variable(t.Tensor(s).cuda())
    data = {}
    A = np.array(result)
    data['A'] = A
    scio.savemat('D:/matlab_ac_control_list.mat',{'A':data['A']})
    return

############################ Now for main! ##############################
def main():
    actor = pre_train_for_actor()
    critic = pre_train_for_critic()
    for i in range(4):
        actor, critic = one_iteration(actor, critic)
    result_output(actor)
    return

if __name__ == '__main__':
    main()
        
                 
    
    
        

         