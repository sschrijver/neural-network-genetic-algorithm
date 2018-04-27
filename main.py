"""Entry point to evolving the neural network. Start here."""
import logging
from tqdm import tqdm
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from helpers.evolution_optimizer import EvolutionOptimizer


# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


def train_estimators(estimators, x_train, y_train):
    """Train each estimator.

    Args:
        estimators (list): Current population of estimators.
        x_train (list): Input labels to use for training/evaluating,
        y_train (list): Output labels to use for training/evaluating,
    """
    pbar = tqdm(total=len(estimators))
    for estimator in estimators:
        estimator.fit(x_train, y_train)
        pbar.update(1)
    pbar.close()


def calculate_accuracy_estimators(estimators, x_test, y_test):
    """Calculate accuracy for each estimator.

    Args:
        estimators (list): Current population of estimators.
        x_test (list): Input labels to use for testing/accuracy,
        y_test (list): Output labels to use for testing/accuracy,
    """
    pbar = tqdm(total=len(estimators))
    for estimator in estimators:
        estimator.score(x_test, y_test)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(estimators):
    """Get the average accuracy for a group of networks.

    Args:
        estimators (list): List of estimators

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for estimator in estimators:
        total_accuracy += estimator.accuracy

    return total_accuracy / len(estimators)


def generate(generations, population, param_choices, x_train, x_test, y_train, y_test):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of estimators in each generation
        param_choices (dict): Parameter choices for estimators
        dataset (str): Dataset to use for training/evaluating

    """
    evolution_optimizer = EvolutionOptimizer(param_choices)
    estimators = evolution_optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***", (i + 1), generations)

        # Train estimators.
        train_estimators(estimators, x_train, y_train)

        # Get accuracy
        calculate_accuracy_estimators(estimators, x_test, y_test)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(estimators)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %d %%", (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            estimators = evolution_optimizer.evolve(estimators)

    # Sort our final population.
    estimators = sorted(estimators, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 estimators.
    print_estimators(estimators[:5])


def print_estimators(estimators):
    """Print a list of networks.

    Args:
        estimators (list): The population of estimators.

    """
    logging.info('-'*80)
    for estimator in estimators:
        estimator.print_estimator()


def get_dataset():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, random_state=42, test_size=0.2)


def main():
    """Evolve an estimator."""
    generations = 50  # Number of times to evolve the population.
    population = 50  # Number of estimators in each generation.

    x_train, x_test, y_train, y_test = get_dataset()

    print(x_train.shape)

    param_choices = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_child_weight': [1, 2, 3, 4],
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.1, 0.2, 0.4, 0.6, 0.8, 1],
    }

    logging.info("***Evolving %d generations with population %d***",
                 generations, population)

    generate(generations, population, param_choices, x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
