# There is a big problem with the first implementation of the Neuron class : 
# Since the instantiation uses a weight parameter, for every weight you need 
# to create an object, which is not the best way to do it since we are creating 
# a lot of objects in each loop. Even if the language use some kind of garbage 
# collection, it's not a very elegant solution. 

inputs = [1,2,3,4]
targets = [3,6,9,12]


class Neuron: # implementation of a Neuron 
    def __init__(self): # this neuron is memoryless 
        pass
    
    def error(self,weight,input,target): # implements cost function 
        prediction = weight * input
        return prediction - target


if __name__ == '__main__':
    weight = 0.01
    neuron = Neuron()    
    learning_rate = 0.01
    epochs = 300

    #training network
    # at this code, the weight is not self-optimizing yet 
    for x in range(0,epochs):
        sample_errors = []

        for sample in zip(inputs,targets):
            sample_errors.append(neuron.error(weight,sample[0],sample[1]))

        avg = sum(sample_errors) / len(sample_errors)
        weight -= learning_rate * avg
        print(neuron.error(weight,sample[0],sample[1]))