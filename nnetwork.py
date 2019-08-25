import numpy as np
import dill

### NETWORK CLASSES

class nNetwork(object):
    def __init__(self, numX, numY, hlayers, activation, cost=(lambda Y, o : Y - o)):
        # Define activation function. 
        # Expects 'activation' to be a class with forward() and backward() static methods
        self.activation = activation
        
        # Expects a function that can take labels, outputs, and return error
        self.cost = cost
        
        # Initialize network layers as a list of nlayer objects
        # Expects 'hlayers' to be a list of integers; each being the size of each layer
        self.layer = [ nLayer(i, activation) for i in hlayers ]
        self.layer.insert(0, nLayer(numX, activation))
        self.layer.append(nLayer(numY, activation))
        
        # Initialize WEIGHT MATRICIES and BIASES for each layer
        # WEIGHT MATRICIES are the theta multipliers applied to the
        # PREVIOUS layer's output to determine the CURRENT layer's activations
        # Each node is represented by a row; each input is a column
        for i in range(1, len(self.layer)):
            prevnodes = self.layer[i-1].len
            curnodes = self.layer[i].len
            # Add an additional set of weights to serve as biases for each node in this layer
            self.layer[i].w = np.random.randn(prevnodes + 1, curnodes)
    
    def forward(self, X):
        # Input layer activation
        a = self.layer[0].input_activation(X)
        
        # Hidden and output layer activation
        for l in self.layer[1:]:
            a = l.activation(a)
        
        # Clean bias unit from output layer
        self.layer[-1].a = np.delete(self.layer[-1].a, 0, axis=1) 
        return a
    
    def backward(self, Y, rate):
        prime = self.activation.backward
        cost = self.cost
        
        # Calculate delta for all layers
        for l in self.layer[::-1]:
            if l == self.layer[-1]:
                # For output layer:
                # Calculate error (labels - predictions)
                # Calculate delta
                l.error = cost(Y, l.a)
                l.delta = l.error*prime(l.a)
                pre_d = l.delta
                pre_w = l.w
                
            elif l == self.layer[0]:
                # For input layer:
                # Do nothing
                pass
            
            else:
                # For hidden layers:
                # Calculate error (apply succeeding layer's delta to adjoined weights)
                # Calculate delta by applying error to derivative of layer's output
                l.error = pre_d @ pre_w.T[:, 1:]    # Don't backpropogate from the Bias node!
                l.delta = l.error*prime(l.a[:, 1:]) # Don't backpropogate from the Bias node!
                pre_d = l.delta
                pre_w = l.w
        
        # Apply delta for all layers
        for l in self.layer:
            if l == self.layer[0]:
                # For input layer:
                # Do nothing
                pre_a = l.a
            else:
                # For hidden & output layers:
                # Set weights to weights + deltas * inputs * learning rate
                l.w += (pre_a.T @ l.delta) * rate
                pre_a = l.a          
                     
    def train(self, X, Y, epochs, rate=1, margin=0):
        ssq = 1
        
        print(f'Training for {epochs} epochs at α={rate}:')
        for i in range(0,epochs):
            a = self.forward(X)
            self.backward(Y, rate)
            if abs(np.mean(np.square(Y - a)) - ssq) > margin:
                ssq = np.mean(np.square(Y - a))
                print(f'Epoch {i+1} SSq Loss: {ssq}')
            else:
                sq = np.mean(np.square(Y - a))
                print(f'Epoch {i+1} SSq Loss: {ssq}')
                print(f'Absolute margin of {margin} has been attained!\nTraining Complete!')
                return True
        print(f'Training concluded without reaching an abslute margin of {margin}...')
        return False    
    
    def describe(self):
        # Descriptive tool for displaying network's size, shape, etc.
        out = ''
        
        print(f'Input Nodes (X): {self.layer[0].len}')
        print(f'Output Nodes (Y): {self.layer[-1].len}')
        print(f'Hidden Layers: {len(self.layer[1:-1])} Total')
        print(f'   Sizes: {[l.len for l in self.layer[1:-1]]}')
        print('\nNetwork Shape:')
        for index, item in enumerate(self.layer):
            if index == 0:
                out += str(item.len) + 'X '
            elif index == len(self.layer)-1:
                out += '| ' + str(item.len) + 'Y'
            else:
                out += '| ' + str(item.len) + ' '
        print(out)
        
    def export(self, name="nNetwork"):
        with open(name+'.dill', 'wb') as file:
            dill.dump(self, file, dill.HIGHEST_PROTOCOL)

    def predict(self, X):
        print(f'Predictions based on input array:\n{X}')
        return self.forward(X)

class nLayer(object):
    def __init__(self, nodecount, activation):
        self.len = nodecount         # Number of nodes/perceptrons. Used for reports. 
        self.w = None                # Weight matrix (declared in nNetwork object)
        self.z = None                # (Dot product of inputs x weight matrix) + bias'
        self.a = None                # 1D array of unit outputs
        self.error = None            # Layer error
        self.delta = None            # 1D array of deltas, for backprop
        self.g = activation.forward  # Layer activation function
        
    def activation(self, i):
        self.z = i @ self.w
        self.a = self.g(self.z)
        self.a = np.hstack((np.ones((np.size(self.a,0),1)), self.a))
        return self.a
    
    def input_activation(self, i):
        self.z = i
        self.a = self.z
        self.a = np.hstack((np.ones((np.size(self.a,0),1)), self.a))
        return self.a
    
    def describe(self):
        print(f'Layer Nodes: {self.len}')
        print(f'\nw (input weights): {len(self.w)} (inputs) x {len(self.w.T)} (nodes)\n{self.w}')
        print(f'\nz (weighted sums): {len(self.z)}\n{self.z}')
        print(f'\na (outputs): {len(self.a)}\n{self.a}')
        print(f'\nerrors: {len(self.error)}\n{self.error}')
        print(f'\ndeltas: {len(self.delta)}\n{self.delta}')
        print(f'\nactivation function: {self.g}')


### ACTIVATION FUNCTIONS (CLASSES)       

class sigmoid(object):
    @staticmethod
    def forward(x):
        return 1/(1+np.exp(-x))
    
    # TECHINICALLY, sigmoid' is sigmoid(x) * (1 - sigmoid(x))
    # However, we are applying this to the nNetwork.layer.a values, 
    # which have already had the sigmoid function applied to them.
    # Therefore, we will use x * (1 - x) for backpropogation.
    @staticmethod
    def backward(x):
        return x * (1 - x)

### COST FUNCTIONS

def crossentropy(Y, o):
    # Commonly used cost funtion for classification problems
    if Y == 1:
        return -(np.log(o))
    else:
        return -(np.log(1 - o))
     
### SUPPORT FUNCTIONS

def nN_scale(X, Y, Xmax=None, Ymax=None):
    # Takes Training Set array X and Testing Set array Y; returns adjusted values for more efficient computation.
    if not Xmax:
        Xmax = np.amax(X, axis=0)
    if not Ymax:
        Ymax = np.amax(Y, axis=0)
    
    X_b = X/Xmax
    Y_b = Y/Ymax
    
    return X_b, Y_b

def nN_import(path):
    # Support function for importing a trained network object using Dill
    with open(path, 'rb') as file:
        return dill.load(file)