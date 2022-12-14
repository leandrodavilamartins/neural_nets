inputs = [1,2,3,4]
targets = [3,6,9,12]


class Neuron: # implementation of a Neuron 
    def __init__(self,weight):
        self.weight = weight
    
    def predict(self, input):
        result = input * self.weight
        return result
    
    def error(self,input,target): # implements cost function 
        prediction = self.predict(input)
        return prediction - target


if __name__ == '__main__':
    weight = 2.1
    neuron = Neuron(weight)    

    for sample in zip(inputs,targets):
        print(neuron.error(sample[0],sample[1]))