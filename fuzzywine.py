import random
from evocomp import EvoComp
from forwardnet import ForwardNet


def fuzzy_wine():
    """
    Uploads fuzzy data and implements ec to try and find the best network
    :return: None
    """
    training_file = 'fuzzy_wine_inputs.csv'  # 22 inputs, 13 outputs

    num_inputs = 22
    num_outputs = 13

    training_data = ForwardNet.load_training_data(training_file, num_inputs)
    random.shuffle(training_data)  # Shuffle the training data to keep results useful
    num_samples = 750  # Number of samples from the training data to use (reduces amount of computation)
    init_population_size = 50  # Number of parents to start out with
    selection_size = 50  # Number of survivors between the parents and the offspring (mu + lambda)
    generations = 15  # Iterations

    min_weight = -1  # Min value for weights and biases
    max_weight = 1  # Max value for weights and biases
    min_hidden = 5  # Min number of hidden nodes in a layer
    max_hidden = 10  # Max number of hidden nodes in a layer
    min_layer = 1  # Min number of layers
    max_layer = 3  # Max number of layers
    mut_chance = 50  # Mutation chance for each individual weight

    ec = EvoComp(init_population_size=init_population_size, selection_size=selection_size, generations=generations,
                 training_data=training_data, num_samples=num_samples, num_inputs=num_inputs, num_outputs=num_outputs,
                 min_weight=min_weight, max_weight=max_weight, min_hidden=min_hidden, max_hidden=max_hidden,
                 min_layer=min_layer, max_layer=max_layer, input_type='raw_input', output_type='raw_output',
                 type_init='select', mut_chance=mut_chance, mutate_data=False)

    ec.evolutionary_computation()


if __name__ == '__main__':
    fuzzy_wine()