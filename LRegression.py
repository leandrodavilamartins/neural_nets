# Linear regression using a Neuron instance and plotting with matplotlib 

import matplotlib.pyplot as plt
import numpy as np

inputs = [1,2,3,4,5,10]
targets = [3.2,6.3,8,13,14,10]


class Neuron: # implementation of a Neuron 
    def __init__(self): # this neuron is memoryless 
        self.weight = 0
        
    
    def error(self,weight,input,target): # implements cost function 
        prediction = weight * input
        return prediction - target


if __name__ == '__main__':
    weight = 0.01
    neuron = Neuron()    
    learning_rate = 0.01
    epochs = 300
    stop = 40

    #training network
    # at this code, the weight is not self-optimizing yet 
    for x in range(0,epochs):
        sample_errors = []

        for sample in zip(inputs,targets):
            sample_errors.append(neuron.error(weight,sample[0],sample[1]))

        avg = sum(sample_errors) / len(sample_errors)
        weight -= learning_rate * avg
        #print(neuron.error(weight,sample[0],sample[1]))
    

    def image(weight,x):
        y = weight * x + 3 # 3 is the linear coefficient ( or bias )
        return y

    x_points = np.arange(0,stop)
    y_points = []
    
    for i in x_points:
        y = image(weight,i)
        y_points.append(y)
    
    #print(x_points)
    #print(y_points)

    neuron.weight = weight
    print(neuron.weight)

    plt.plot(targets, 'o')
    plt.plot(x_points,y_points) 
    plt.show()
