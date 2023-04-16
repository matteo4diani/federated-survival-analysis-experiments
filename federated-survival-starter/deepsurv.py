import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from DeepSurv.loss import cox_ph_loss
from sksurv.metrics import concordance_index_censored


class SurvNet(nn.Module):
    def __init__(self, net_dims, activation, dropout_p, batch_norm):
        super(SurvNet, self).__init__()
        
        self.net_dims = net_dims
        self.layers = nn.ModuleList()
        if batch_norm == True:
            self.batch_norm = [True]*len(self.net_dims)
        elif batch_norm == False: 
            self.batch_norm = [False]*len(self.net_dims)
        else:
            self.batch_norm = batch_norm
        
        
        # Add activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'linear':
            self.activation = 'linear'
        else:
            raise ValueError("Invalid activation function")
            
            
        # Add dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Add encoding layers
        num_layers = len(self.net_dims) - 1
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.net_dims[i], self.net_dims[i+1]))
            if self.batch_norm[i]:
                self.layers.append(nn.BatchNorm1d(self.net_dims[i+1]))
        
        self.train_history_ = {'learn_curve_train' : [], 'cindex_train' : []}
        self.eval_history_ = {'learn_curve_test' : [], 'cindex_test' : []}
    
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.activation != 'linear':
                x = self.activation(x)
            x = self.dropout(x)
        
        risk = torch.exp(x)
        return risk
        
    
    def predict_risk(self, x):
        return self.forward(x)
        
    
    def compute_loss(self, r,t,e):
        loss_cox = cox_ph_loss(r, t, e)
        return loss_cox
    
    def evaluate_cindex(self, data):
        X, status, time = data[0], data[1].detach().numpy(), data[2].detach().numpy()
        risk = self.predict_risk(X).detach().numpy().flatten()
        CI = CI_tr=concordance_index_censored(status, time, risk)[0]
        
        return(CI)
        
    
    
    def fit(self, train_loader, test_loader, optimizer, num_epochs, verbose=True):

        # Train
        for epoch in range(num_epochs):
            self.train()
            for x,event,time,idx in train_loader:

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                risk = self.forward(x)
                loss = self.compute_loss(risk, time, event)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            if test_loader is not None:
                self.eval()
                with torch.no_grad():
                    for x,event,time,idx in test_loader:
                        # Forward pass
                        risk = self.forward(x)
                        loss_test = self.compute_loss(risk, time, event)
                        
                CI_test = self.evaluate_cindex(test_loader.dataset.tensors)
                self.eval_history_['cindex_test'].append(CI_test)
                
            else: 
                loss_test = torch.tensor(torch.inf)
            
            self.train_history_['learn_curve_train'].append(loss.item())
            self.eval_history_['learn_curve_test'].append(loss_test.item())
            
            CI_train = self.evaluate_cindex(train_loader.dataset.tensors)            
            self.train_history_['cindex_train'].append(CI_train)

            
            if verbose:
                # Print the loss every 10 epochs
                if (epoch+1) % 10 == 0:
                    print ('Train: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
                    print ('Test: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss_test.item()), '\n')

        return self