import numpy as np
import random

from settings import POPULATION, SELECTION_PERCENTAGE, CROSSOVER_RATE, MUTATION_RATE
from settings import INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER


class Neural:

    def __init__(self, genome=None):

        # One Hidden Layer Model consists of (3, 6, 1) plus bias terms

        self.INPUT_LAYER = INPUT_LAYER
        self.HIDDEN_LAYER = HIDDEN_LAYER
        self.OUTPUT_LAYER = OUTPUT_LAYER

        self.w1 = np.random.uniform(-1,1,(self.HIDDEN_LAYER, self.INPUT_LAYER + 1))
        self.w2 = np.random.uniform(-1,1,(self.OUTPUT_LAYER,self.HIDDEN_LAYER + 1))

        if genome is not None:

            self.decode(genome)

    def __sigmoid(self, z):

        return 1/(1 + np.exp(-1*z))

    def __regularize(self, input):

        sumVal = np.sum(abs(input))

        for val in input:

            if sumVal == 0:
                val[0] = 0
            else:
                val[0] = val[0]/sumVal

    # Input must be of shape (3,1)
    def predict(self,input):

        # Regularize  the Input for val to be between -1 to 1
        self.__regularize(input)

        # Add bias term for functionality changes shape to (4,1)
        input_new = np.concatenate((np.array([[1]]), input), axis = 0)
        hidden_layer = self.w1@input_new

        hidden_layer = self.__sigmoid(hidden_layer)
        hidden_layer_new =  np.concatenate((np.array([[1]]), hidden_layer), axis = 0)

        output_layer = self.w2@hidden_layer_new

        out = self.__sigmoid(output_layer)

        return [-1] if out <= 0.5 else [1]

    # Convert complete two matrix of weights in single list
    def encode(self):

        flat_list = self.w1.flatten()
        flat_list = flat_list.tolist()

        out_list = self.w2.flatten().tolist()

        flat_list.extend(out_list)

        return flat_list

    # Convert back list to two matrix
    def decode(self, genome):

        for i in range(self.HIDDEN_LAYER):
            for j in range(self.INPUT_LAYER + 1):
                self.w1[i][j] = genome[i*(self.INPUT_LAYER + 1) + j]

        for i in range(self.OUTPUT_LAYER):
            for j in range(self.HIDDEN_LAYER + 1):
                self.w2[i][j] = genome[(i*(self.HIDDEN_LAYER + 1)) + j + self.HIDDEN_LAYER*(self.INPUT_LAYER+1)]

    # This takes Neural object list as input and create new eltie birds
    @classmethod
    def selection(cls,neural_list):

        elite_birds_copy = []
        elite_birds = neural_list[0:round(SELECTION_PERCENTAGE*POPULATION)]

        # Get elite birds and ecode their weights and insert it to copy
        for bird in elite_birds:
            gen = bird.encode()
            elite_birds_copy.append(Neural(gen))

        return elite_birds_copy

    @classmethod
    def mutation(cls, neural_object):

        gen = neural_object.encode()

        for i in range(len(gen)):
            if random.randint(0,100) <= MUTATION_RATE*100:
                gen[i] += np.random.uniform(-0.5,0.5)

        new_object = Neural(gen)

        return new_object

    @classmethod
    def crossover(cls, object1, object2):

        gen1 = object1.encode()
        gen2 = object2.encode()

        for val in range(len(gen1)):
            if random.randint(0,100) <= CROSSOVER_RATE*100:
                gen1[val], gen2[val] = gen2[val], gen1[val]

        return Neural(gen1) if random.randint(0,1) == 0 else Neural(gen2)

        #return [Neural(gen1), Neural(gen2)]

    @classmethod
    def create_new_generation(cls, neural_list):

        new_generation = []

        if len(neural_list) == 0:

            for i in range(POPULATION):
                Obj = Neural(None)
                new_generation.append(Obj)

        else:
            elite_neural = Neural.selection(neural_list)
            new_generation.extend(elite_neural)

            top_units = len(new_generation)
            # Apply Mutation for some birds
            for i in range(top_units, POPULATION):

                offspring = None
                if i == top_units :
                    # new_generation.append(Neural.crossover(new_generation[0],new_generation[1]))
                    offspring = Neural.crossover(elite_neural[0],elite_neural[1])

                elif i < POPULATION - 2:
                    parentA = random.randint(0, len(elite_neural) - 1)
                    parentB = random.randint(0, len(elite_neural) - 1)

                    offspring = Neural.crossover(elite_neural[parentA], elite_neural[parentB])
                    # new_generation.append(Neural.crossover(elite_neural[parentA], elite_neural[parentB]))
                else:
                    parentA = random.randint(0,len(elite_neural) - 1)
                    offspring = elite_neural[parentA]

                offspring = Neural.mutation(offspring)
                new_generation.append(offspring)

            '''
            for i in range(round(MUTATION_RATE*POPULATION)):
                new_generation.append(Neural.mutation(neural_list[i]))
    
            # Apply Crossover
            for i in range(round((MUTATION_RATE * 100 / POPULATION)),
                           round(((MUTATION_RATE * 100 / POPULATION) + (CROSSOVER_RATE * 100 / POPULATION)))):
                new_generation.append(Neural.crossover(neural_list[i], elite_neural[random.randint(0,len(elite_neural) -1)])[0])
    
            for i in range(POPULATION - len(new_generation)):
                new_generation.append(Neural(None))
            '''
        return new_generation
