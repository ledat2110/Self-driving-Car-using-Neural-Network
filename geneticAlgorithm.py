import random

class Genome(object):
	def __init__(self, genes = [], fitness = 0):
		self.fitness = fitness
		self.genes = genes
	
	def __lt__(self, genome):
		return self.fitness < genome.fitness

class GeneticAlgorithm(object):
	def __init__(self, popSize, numOfGenes):
		self.population = []
		self.popSize = popSize
		self.numOfGenes = numOfGenes
		self.totalFitness = 0
		self.bestFitness = 0
		self.worstFitness = 0
		self.avgFitness = 0
		self.fittestGenome = None
		self.mutationRate = 0.3
		self.crossoverRate = 0.6
		self.generationNum = 0
		self.mutationPerc = 10
		
		for i in range(0, popSize):
			self.population.append(Genome())
			
			for j in range(0, self.numOfGenes):
				self.population[i].genes.append(random.uniform(-1.0, 1.0))
	
	def mutate(self, genome):
		for i in range(0, len(genome.genes)):
			if random.uniform(0.0, 1.0) < self.mutationRate:
				genome.genes[i] += random.uniform(-2.0, 2.0)
		return genome
    	
	
	def getGenomeByTournament(self):
		num = random.randint(0, len(self.population) - 1)
		fittest = self.population[num]
		
		for i in range(0, self.popSize):
			if self.population[i].fitness > fittest.fitness:
				fittest = self.population[i]
		
		return fittest;
	
	def crossover(self, parent1, parent2):
		child1 = Genome()
		child2 = Genome()
		for i in range(0, len(parent1.genes)):
			if random.uniform(0.0, 1.0) < self.crossoverRate:
					child1.genes[i] = parent2.genes[i]
					child2.genes[i] = parent1.genes[i]
			else:
					child1.genes[i] = parent1.genes[i]
					child2.genes[i] = parent2.genes[i]
		return child1, child2

	
	def calcFitness(self):
		self.totalFitness = 0
		self.bestFitness = 0
		self.worstFitness = 0
		self.avgFitness = 0
		
		self.worstFitness = self.population[0].fitness
		for i in range(0, self.popSize):
			self.totalFitness += self.population[i].fitness
			if self.population[i].fitness >= self.bestFitness:
				self.bestFitness = self.population[i].fitness
				self.fittestGenome = self.population[i]
			if self.population[i].fitness < self.worstFitness:
				self.worstFitness = self.population[i].fitness
		
		self.avgFitness = self.totalFitness / self.popSize
	
	def update(self):
		self.calcFitness()
		
		newPopulation = []
		bestPopulation = self.population[:]
		bestPopulation.sort(reverse=True)
		
		newPopulation.append(bestPopulation[0])
		newPopulation.append(bestPopulation[1])
		
		while len(newPopulation) < self.popSize:
			idx1 = random.randint(0, len(newPopulation) - 1)
			idx2 = random.randint(0, len(newPopulation) - 1)
			while(idx2 == idx1):
				idx2 = random.randint(0, len(newPopulation) - 1)
			
			parent1 = newPopulation[idx1]
			parent2 = newPopulation[idx2]
			
			child1, child2 = self.crossover(parent1, parent2)
			
			newPopulation.append(child1)
			if len(newPopulation) < self.popSize:
				newPopulation.append(child2)
		
		for i in range(2, self.popSize):
			newPopulation[i] = self.mutate(newPopulation[i])

		self.population = newPopulation
		self.generationNum += 1
		
		for genome in self.population:
			genome.fitness = 0
	
	
