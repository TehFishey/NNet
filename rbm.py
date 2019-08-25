import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class RBM(object):
    def __init__(self, vnodes, hnodes):
        self.weights = torch.randn(hnodes, vnodes)
        self.hbias = torch.randn(1, hnodes)
        self.vbias = torch.randn(1, vnodes)
        
        self.vnodes = vnodes
        self.hnodes = hnodes
        
    def h_sample(self, v):
        wv = torch.mm(v, self.weights.t())
        hz = wv + self.hbias.expand_as(wv)
        
        p_h = torch.sigmoid(hz)
        hsamp = torch.bernoulli(p_h)
        return p_h, hsamp
    
    def v_sample(self, h):
        wh = torch.mm(h, self.weights)
        vz = wh + self.vbias.expand_as(wh)
        
        p_v = torch.sigmoid(vz)
        vsamp = torch.bernoulli(p_v)
        return p_v, vsamp

    def rbm_train(self, training_data, data_size, epochs, steps, batch_size=1):

        print(f'Contrastive Divergence for {epochs} epochs in batches of {batch_size}, with k={steps} :')
        for i in range(0, epochs):
            abs_loss = 0
            x = training_data
            for n in range(0, data_size - batch_size, batch_size):
                # Contrastive Divergence via Gibbs Sampling, working in batches
                vk = x[n:n+batch_size]    # Visible node values after step k for batch
                v0 = x[n:n+batch_size]    # Original Values for batch; training goals
                ph0,_ = self.h_sample(v0)     # Original p(h=1|v)
                for k in range(0, steps):     # 'Blind Walk' Markov Chain (Not random; probablilities are different)
                    _,hk = self.h_sample(vk)  # "Forward" processing, take bernoulli outputs
                    _,vk = self.v_sample(hk)  # "Backward" processing, take bernoulli outputs
                    vk[v0<0] = v0[v0<0]       # Lock -1 (null) values based on orignal values
                phk,_ = self.h_sample(vk)     # Final p(h=1|vk)
                
                # Algorithm from (Fischer & Igel, 2012)
                self.weights += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
                self.vbias += torch.sum((v0-vk), 0)
                self.hbias += torch.sum((ph0-phk), 0)
                
            abs_loss += torch.mean(torch.abs(v0[v0 > 0] - vk[v0 > 0])) # Don't factor nulls in; the model always gets them right
            print(f'Epoch {i+1} Mean Absolute Loss (Normalized): {abs_loss/(i+1)}')
            
    def rbm_test(self, training_data, testing_data, data_size):
        abs_loss = 0
        diff_count = 0
        for n in range(0, data_size): 
            vpred = training_data[n:n+1]   # Inputs to create predictions for test-set only values
            vtarg = testing_data[n:n+1]    # Full list of values, including ones omitted from training set
            if len(vtarg[vtarg>=0]) > 0:
                _,hpred = self.h_sample(vpred)
                _,vpred = self.v_sample(hpred)
                abs_loss += torch.mean(torch.abs(vtarg[vtarg > 0] - vpred[vtarg > 0]))
                diff_count += 1
                print(f'Test Loss for userid {n}: {abs_loss/diff_count} (Mod count {diff_count}) (abs {abs_loss})')
        print(f'Testing with single step...')
        print(f'Mean Absolute Test Loss (Normalized): {abs_loss/diff_count}')
