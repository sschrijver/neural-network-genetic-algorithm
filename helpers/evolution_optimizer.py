""""Docstring placeholder"""
from functools import reduce
from operator import add
import random
from models.estimator import Estimator


class EvolutionOptimizer():
    """
    Class that holds a genetic algorithm for evolving an estimator.
    Credit:
        A lot of those code was originally inspired by:
        http://lethain.com/genetic-algorithms-cool-name-damn-simple/
    """
    def __init__(self, param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.
        Args:
            param_choices (dict): Possible estimator paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected estimator
                remaining in the population
            mutate_chance (float): Probability an estimator will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.param_choices = param_choices

    def create_population(self, count):
        """Create a population of random estimators.
        Args:
            count (int): Number of estimators to generate, aka the
                size of the population
        Returns:
            (list): Population of estimator objects
        """
        pop = []
        for _ in range(0, count):
            # Create a random estimator.
            estimator = Estimator(self.param_choices)
            estimator.create_random()

            # Add the estimator to our population.
            pop.append(estimator)

        return pop

    @staticmethod
    def fitness(estimator):
        """Return the accuracy, which is our fitness function."""
        return estimator.accuracy

    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            (float): The average accuracy of the population
        """
        summed = reduce(add, (self.fitness(estimator) for estimator in pop))
        return summed / float((len(pop)))

    def breed(self, mother: Estimator, father: Estimator):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """
        children = []
        for _ in range(2):

            child = {}

            for param in self.param_choices:
                child[param] = random.choice(
                    [mother.parameters[param],
                     father.parameters[param]]
                )

            # Now create a network object.
            estimator = Estimator(self.param_choices)
            estimator.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                estimator = self.mutate(estimator)

            children.append(estimator)

        return children

    def mutate(self, estimator: Estimator):
        """Randomly mutate one part of the network.
        Args:
            estimator (Estimator): The estimator to mutate
        Returns:
            (Estimator): A randomly mutated estimator object
        """
        # Choose a random key.
        mutation = random.choice(list(self.param_choices.keys()))

        # Mutate one of the params.
        estimator.parameters[mutation] = random.choice(self.param_choices[mutation])

        return estimator

    def evolve(self, pop: [Estimator]):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        # Get scores for each network.
        graded = [(self.fitness(estimator), estimator)
                  for estimator in pop]

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
