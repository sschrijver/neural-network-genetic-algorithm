from functools import reduce
from operator import add
import random
from geneticmodel import GeneticModel

class Optimizer():
    def __init__(self, param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.param_choices = param_choices

    def create_population(self, count):
        pop = []
        for _ in range(0, count):
            # Create a random network.
            genetic_model = GeneticModel(self.param_choices)
            genetic_model.create_random()

            # Add the network to our population.
            pop.append(genetic_model)

        return pop

    @staticmethod
    def fitness(genetic_model):
        return genetic_model.accuracy

    def grade(self, pop):
        summed = reduce(add, (self.fitness(genetic_model) for genetic_model in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        children = []
        for _ in range(2):

            child = {}

            for param in self.param_choices:
                child[param] = random.choice(
                    [mother.genetic_model_settings[param],
                     father.genetic_model_settings[param]]
                )

            # Now create a network object.
            genetic_model = GeneticModel(self.param_choices)
            genetic_model.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                genetic_model = self.mutate(genetic_model)

            children.append(genetic_model)

        return children

    def mutate(self, genetic_model):
        # Choose a random key.
        mutation = random.choice(list(self.param_choices.keys()))

        # Mutate one of the params.
        genetic_model.genetic_model_settings[mutation] = random.choice(self.param_choices[mutation])

        return genetic_model

    def evolve(self, pop):
        # Get scores for each network.
        graded = [(self.fitness(genetic_model), genetic_model) 
                  for genetic_model in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents