from NeuralNet import NeuralNet
import random
import time

#Results are only visible when the seed is constant throughout the algorithm
random.seed(5)

class NeuroEvolution:

    population = []
    newPopulation = []
    populationSize = 0
    maxLayers = 0
    maxNodes = 0
    maxIterations = 0
    threshold = 0
    activations = ['relu', 'sigmoid', 'tanh']
    trend = []

    def __init__(self, populationSize = 10, maxLayers = 7, maxNodes = 16, maxIterations = 5, threshold = 1):
        self.populationSize = populationSize
        self.maxLayers = maxLayers
        self.maxNodes = maxNodes
        self.maxIterations = maxIterations
        self.threshold = threshold

    def initialize(self):
        for i in range(self.populationSize):
            self.newPopulation.append(NeuralNet(random.randint(1, self.maxLayers), random.randint(1, self.maxNodes), self.activations[random.randint(0, len(self.activations) - 1)]))

    def select(self):
        pass

    def mutate(self, nn):
        chance = random.randint(1, 1000)
        if(chance > 990):
            nn.layers = random.randint(1, self.maxLayers)
        chance = random.randint(1, 1000)
        if(chance > 990):
            nn.nodes = random.randint(1, self.maxNodes)
        chance = random.randint(1, 1000)
        if(chance > 990):
            nn.activation = self.activations[random.randint(0, len(self.activations) - 1)]
        return nn

    def crossover(self, index1, index2):
        randLayers = index1 if random.randint(0, 2) == 0 else index2
        randNodes = index1 if random.randint(0, 2) == 0 else index2
        randActivation = index1 if random.randint(0, 2) == 0 else index2

        return NeuralNet(self.population[randLayers].layers, self.population[randNodes].nodes, self.population[randActivation].activation)


    def trainAll(self, trainX, trainY, epoch):
        for nn in self.newPopulation:
            nn.train(trainX, trainY, epoch)

    def setAllFitness(self, testX, testY):
        for nn in self.newPopulation:
            nn.setFitness(testX, testY)

    def printStatus(self, i, execTime):
        print("-------Iteration " + str(i))
        print("\n\tTime = " + str(execTime))
        print("\tBest Accuracy = " + str(self.population[0].getAccuracy()))
        print("\tBest Loss = " + str(self.population[0].getLoss()))
        print("-------------------\n")
        for individual in self.population:
            print(individual.getAccuracy())
        input()
        self.trend.append(self.population[0].getAccuracy())

    def printBestNet(self, execTime):
        print("\n###############################################################\n")
        print("Total number of iterations = " + str(self.maxIterations))
        print("Total time = " + str(execTime))
        print("Best Accuracy = " + str(self.population[0].getAccuracy()))
        print("Best Loss = " + str(self.population[0].getLoss()))
        print("Number of Layers = " + str(self.population[0].layers))
        print("Number of Nodes = " + str(self.population[0].nodes))
        print("Activation Function = " + str(self.population[0].activation))
        for ind in self.trend:
            print(ind)
        print("\n###############################################################\n")

    def run(self, trainX, trainY, testX, testY, epoch):

        self.initialize()
        start = time.time()
        tempStart = start
        for i in range(self.maxIterations):
            self.trainAll(trainX, trainY, epoch)
            self.setAllFitness(testX, testY)
            self.newPopulation = sorted(self.newPopulation, key = lambda net: net.getAccuracy(), reverse = True)
            self.population = self.newPopulation[:self.populationSize]
            self.newPopulation = []
            self.printStatus(i, time.time() - tempStart)
            tempStart = time.time()
            if self.population[0].getAccuracy() >= self.threshold:
                break
            for j in range(int(self.populationSize/2)):
                parent1 = 0
                parent2 = 0
                while parent1 == parent2:
                    parent1 = random.randint(0, int(self.populationSize/2))
                    parent2 = random.randint(0, int(self.populationSize/2))
                self.newPopulation.append(self.crossover(parent1, parent2))
            self.newPopulation += self.population[:int(self.populationSize/2)]
            for nn in self.newPopulation:
                nn = self.mutate(nn)
        end = time.time()
        self.printBestNet(end - start)
