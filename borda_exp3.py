import numpy as np

from math import log, sqrt


class BordaEXP3Con():

    def __init__(self, K, T, d, B,opt,Z, comparison_function, consumption_gen, true_rew):
        self.T = T
        # self.bandit = bandit
        self.bandit = comparison_function
        self.t = 0
        self.reward = 0
        self.rewards = []
        self.rew_fn = true_rew
        self.K = K
        self.cnsmp_func = consumption_gen
        
        
        self.eta = 0.5
        self.gamma = 0.25
        self.q = np.ones(K)*1/self.K
        self.s = np.zeros(K)
        self.u = np.zeros(K)
        self.l = np.zeros(K)
        self.lamdba = np.ones(d)*1/d
        self.Z = Z
        self.B = B
        self.opt = opt
        self.cnsmp = 0
        self.d = d
    def algo(self):
        #print(self.q)
        x = np.random.choice(np.arange(self.K), p=self.q)
        y = np.random.choice(np.arange(self.K), p=self.q)
        o_t = self.bandit(x,y)
        u_t,v_t = self.cnsmp_func(x,y)
        
        #print(o_t,u_t,v_t)
        
        #for i in range(K):
        self.s[x] = 1/(self.K*self.q[x])* o_t/self.q[y]
        self.u[x] = 1/self.q[x] * u_t
        self.u[y] = 1/self.q[y] * v_t
        
        self.l[x] = 2*self.s[x] + self.Z * np.dot(self.lamdba,np.ones(self.d)*self.B/self.T - (self.u[x] + self.u[x])) 
        
        for i in range(self.K):
            #print("l[i]:", self.l[i])
            self.q[i] = self.q[i]* np.exp(self.eta*self.l[i])
            '''if np.isnan(self.q[i]):
                self.q = np.zeros(self.K)
                self.q[i] = 1
                break'''
                
        #print(self.q)
        #print("******************")
        self.q = self.q/np.linalg.norm(self.q,1)
        
        self.q = (1-self.gamma)*self.q + self.gamma/self.K
        
        #print(self.q)
        
        for i in range(self.d):
            self.lamdba[i] *= np.exp(self.eta* ((self.u[i] + self.u[i]) - self.B/self.T))
        
        self.lamdba /= np.linalg.norm(self.lamdba,1)
        
        self.t += 1
        
        self.reward += self.rew_fn(x,y)
        self.rewards.append(float(self.reward))
        
        self.cnsmp += u_t + v_t
        
        
        return self.rewards, self.cnsmp
        
        
class BordaEXP3():

    def __init__(self, K, T, d, B,opt,Z, comparison_function, consumption_gen, true_rew):
        self.T = T
        # self.bandit = bandit
        self.bandit = comparison_function
        self.t = 0
        self.reward = 0
        self.rewards = []
        self.rew_fn = true_rew
        self.K = K
        self.cnsmp_func = consumption_gen
        
        
        self.eta = 0.5
        self.gamma = 0.25
        self.q = np.ones(K)*1./K
        self.s = np.zeros(K)
        self.u = np.zeros(K)
        self.l = np.zeros(K)
        self.lamdba = np.ones(d)*1/d
        self.Z = Z
        self.B = B
        self.opt = opt
        self.cnsmp = 0
        
    def algo(self):
        x = np.random.choice(np.arange(self.K), p=self.q)
        y = np.random.choice(np.arange(self.K), p=self.q)
        o_t = self.bandit(x,y)
        u_t,v_t = self.cnsmp_func(x,y)
        
        #for i in range(self.K):
        self.s[x] = 1/(self.K*self.q[x])* o_t/self.q[y]

        
        
        for i in range(self.K):
            self.q[i] = self.q[i]* np.exp(self.eta*self.s[x])
        
        self.q = self.q * 1./(np.sum(self.q))
        
        self.q = (1-self.gamma)*self.q + self.gamma/self.K
        
        
        self.t += 1
        
        self.reward += self.rew_fn(x,y)
        self.rewards.append(float(self.reward))
        
        self.cnsmp += u_t + v_t
        
        return self.rewards, self.cnsmp


class BordaTS():

    def __init__(self, K, T, d, B,opt,Z, comparison_function, consumption_gen, true_rew):
        self.T = T
        # self.bandit = bandit
        self.bandit = comparison_function
        self.t = 0
        self.reward = 0
        self.rewards = []
        self.rew_fn = true_rew
        self.K = K
        self.cnsmp_func = consumption_gen
        
        
        self.eta = 0.1
        self.gamma = 0.1
        self.q = np.ones(K)*1./K
        self.s = np.zeros(K)
        self.u = np.zeros(K)
        self.l = np.zeros(K)
        self.lamdba = np.ones(d)*1/d
        self.Z = Z
        self.B = B
        self.opt = opt
        self.cnsmp = 0
        
        #############################
        
        self.alpha = 0.1
        self.unif = np.ones(K)*1./K
        
        self.count = np.zeros((10,10))
        
    def algo(self):
    
        b = np.random.binomial(1,self.alpha,1)[0]
        
        if b==1:
            x = np.random.choice(np.arange(self.K), p=self.unif)
            y = np.random.choice(np.arange(self.K), p=self.unif)
        else:
            beta_samples = np.ones((self.K, self.K))*0.5
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    sampled = np.random.beta(self.count[i][j] + 1, self.count[j][i] + 1)
                    beta_samples[i][j] = sampled
                    beta_samples[j][i] = 1 - sampled
            x = np.argmax(np.sum(beta_samples,axis=1))


            beta_samples = np.ones((self.K, self.K))*0.5
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    sampled = np.random.beta(self.count[i][j] + 1, self.count[j][i] + 1)
                    beta_samples[i][j] = sampled
                    beta_samples[j][i] = 1 - sampled
            y = np.argmax(np.sum(beta_samples,axis=1))    
    
        o_t = self.bandit(x,y)
        u_t,v_t = self.cnsmp_func(x,y)
        
        if o_t == 1:
            self.count[x][y]+= 1
        else:
            self.count[y][x]+= 1
    
        self.reward += self.rew_fn(x,y)
        self.rewards.append(float(self.reward))
        
        self.cnsmp += u_t + v_t
        
        return self.rewards, self.cnsmp  
        
class BordaUCB():

    def __init__(self, K, T, d, B,opt,Z, comparison_function, consumption_gen, true_rew):
        self.T = T
        # self.bandit = bandit
        self.bandit = comparison_function
        self.t = 0
        self.reward = 0
        self.rewards = []
        self.rew_fn = true_rew
        self.K = K
        self.cnsmp_func = consumption_gen
        
        
        self.eta = 0.1
        self.gamma = 0.1
        self.q = np.ones(K)*1./K
        self.s = np.zeros(K)
        self.u = np.zeros(K)
        self.l = np.zeros(K)
        self.lamdba = np.ones(d)*1/d
        self.Z = Z
        self.B = B
        self.opt = opt
        self.cnsmp = 0
        
    def algo(self):
        x = np.random.choice(np.arange(self.K), p=self.q)
        y = np.random.choice(np.arange(self.K), p=self.q)
        o_t = self.bandit(x,y)
        u_t,v_t = self.cnsmp_func(x,y)
        
        #for i in range(self.K):
        self.s[x] = 1/(self.K*self.q[x])* o_t/self.q[y]

        
        
        for i in range(self.K):
            self.q[i] = self.q[i]* np.exp(self.s[x])
        
        self.q = self.q * 1./(np.sum(self.q))
        
        self.q = (1-self.gamma)*self.q + self.gamma/self.K
        
        
        self.t += 1
        
        self.reward += self.rew_fn(x,y)
        self.rewards.append(float(self.reward))
        
        self.cnsmp += u_t + v_t
        
        return self.rewards, self.cnsmp        


