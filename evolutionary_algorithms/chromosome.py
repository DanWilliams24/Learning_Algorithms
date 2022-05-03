class Chromosome:
    # static variable used to keep track of chromosomes over lifetime of evolutionary algorithm
    HISTORICAL_NUMBER = 0

    def __init__(self, bitstring="000000"):
        self.genes = bitstring
        self.historical_marker = Chromosome.HISTORICAL_NUMBER
        # update historical number for newer individuals
        Chromosome.HISTORICAL_NUMBER += 1

    def get_fitness(self):
        fitness = 0
        for i in range(len(self.genes)):
            if self.genes[i] == "1":
                fitness += 1

        return fitness
    
    def get_chromosome(self):
        return self.genes


    def get_gene(self, index):
        return self.genes[index]

    def __len__(self):
        return len(self.genes)
