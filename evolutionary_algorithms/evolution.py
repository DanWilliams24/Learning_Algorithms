import random


class Chromosome:
    # static variable used to keep track of chromosomes over lifetime of evolutionary algorithm
    HISTORICAL_NUMBER = 0

    def __init__(self, bitstring="000000"):
        # print(bitstring.split())
        self.genes = [bit for bit in bitstring]
        self.historical_marker = Chromosome.HISTORICAL_NUMBER
        # update historical number for newer individuals
        Chromosome.HISTORICAL_NUMBER += 1


    def get_fitness(self):
        fitness = 0
        for i in range(len(self.genes)):
            # print(type(self.genes[i]))
            if (i % 2 == 0) and self.genes[i] == "1":
                fitness += 1

            if (i % 2 != 0) and self.genes[i] == "1":
                fitness -= 1

        return fitness

    def get_chromosome(self):
        return self.genes

    def get_gene(self, index):
        return self.genes[index]

    def __len__(self):
        return len(self.genes)

    def __str__(self) -> str:
        return "".join(str(bit) for bit in self.genes) + ":" + str(self.get_fitness())

    def __repr__(self) -> str:
        return "".join(str(bit) for bit in self.genes) + ":" + str(self.get_fitness())


class Evolution:
    def __init__(self, max_pop=8, bitstring_length=6, mutation_prob=0.1, max_generations=50, output_allowed=True):
        # composed of individuals
        self.population = []
        self.max_population = max_pop
        self.gen_count = 0
        self.bitstring_length = bitstring_length
        self.mutation_probability = mutation_prob
        self.max_generations = max_generations
        self.output_allowed = output_allowed

    def init_population(self):
        for i in range(self.max_population):
            bitstring = ""
            for _ in range(self.bitstring_length):
                bitstring += str(random.randint(0, 1))
            self.add_individual(Chromosome(bitstring))

    def evolve(self):
        # create population
        self.init_population()
        self.output(f"Initial Population: {self.population}")
        # keep evolution going until max fitness is achieved
        while self.calc_mean_fitness() <= self.bitstring_length and self.gen_count < self.max_generations:
            self.output(f"Generation {self.gen_count}: {self.population}")
            # performs selection, then crossover, then mutation, then compute fitness
            self.perform_selection()
            self.gen_count += 1

            if self.population[0].get_fitness() >= self.bitstring_length / 2:
                pass
                #print("Max Fitness Reached.")
                return

    def add_individual(self, individual):
        if len(self.population) == self.max_population:
            self.population.pop(self.max_population - 1)

        if len(self.population) == 0:
            self.population.append(individual)
        else:
            inserted = False
            for i in range(len(self.population)):
                if individual.get_fitness() > self.population[i].get_fitness():
                    # print(f"Inserting {individual} at position {i} in {self.population} ")
                    self.population.insert(i, individual)
                    inserted = True
                    break
            if not inserted:
                self.population.append(individual)

    # this selection method only takes the most fit individuals for reproduction
    # pros: Ensures optimal solutions are encouraged
    # cons: if a non-optimal configuration is found to be the most fit in the population,
    # it may suffer from problems converging upon local minima versus the global minimum
    def perform_selection(self):
        # select N fittest individuals 
        # these are O(1) operations since we sort individuals on insertion
        first_fittest = self.population[0]
        second_fittest = self.population[1]

        # crossover N/2 pairs
        self.crossover_individuals(first_fittest, second_fittest)

    def mutate_offspring(self, bitstring):
        for i in range(len(bitstring)):
            choice = random.random()
            # print(f"mutate:{choice} -> {choice >= 1-self.mutation_probability}")
            if choice >= 1 - self.mutation_probability:
                # print(f"modifying {bitstring[i]}")
                if bitstring[i] == "0":
                    bitstring = bitstring[:i] + "1" + bitstring[i + 1:]
                else:
                    bitstring = bitstring[:i] + "0" + bitstring[i + 1:]
            # print(bitstring)
        return bitstring

    def crossover_individuals(self, individualA, individualB):
        # select crossover point, but make sure the point chosen
        # is within the range of the shortest gene length between the two individuals
        crossover_point = random.randint(0, min(len(individualA), len(individualB)))
        offspringA = ""
        offspringB = ""
        # perform crossover
        for i in range(0, crossover_point):
            offspringA += individualB.get_gene(i)
            offspringB += individualA.get_gene(i)

        # persist other part of individuals dna
        for i in range(crossover_point, len(individualA)):
            offspringA += individualB.get_gene(i)

        for i in range(crossover_point, len(individualB)):
            offspringB += individualA.get_gene(i)

        # cause random mutations
        offspringA = self.mutate_offspring(offspringA)
        offspringB = self.mutate_offspring(offspringB)

        # create new individuals and add them to next-gen population
        personA = Chromosome(offspringA)
        personB = Chromosome(offspringB)
        self.add_individual(personA)
        self.add_individual(personB)

    def calc_mean_fitness(self):
        population_size = len(self.population)
        current_sum = 0
        for individual in self.population:
            current_sum += individual.get_fitness()

        return current_sum / population_size

    def output(self, data):
        if self.output_allowed:
            print(data)


def calc_average_convergence(num_epochs, simulation_sample):
    current_sum = 0
    for simulation in simulation_sample:
        current_sum += simulation["generations"]

    return current_sum / num_epochs


if __name__ == '__main__':
    samples = []
    num_epochs = int(input("Enter Number of Epochs to Run: "))
    for i in range(num_epochs):
        genetic_algorithm = Evolution(max_pop=10, bitstring_length=100, mutation_prob=0.05, max_generations=1000,
                                      output_allowed=False)
        genetic_algorithm.evolve()
        print(f"{i}) Mean Fitness: {genetic_algorithm.calc_mean_fitness()} | # of Generations: {genetic_algorithm.gen_count}")
        samples.append({"generations": genetic_algorithm.gen_count,
                        "fitness": genetic_algorithm.calc_mean_fitness()})
    print("Mean Number of Generations: " + str(calc_average_convergence(num_epochs, samples)))




"""
To Do List
- Test varying mutation rate over time
- Convert this problem into a AI Gym Env
- Create a measurement for individual change
- Create a measurement for population change
- Establish good termination condition
- Create an RL model that takes in individual/population change and outputs mutation suggestions

"""