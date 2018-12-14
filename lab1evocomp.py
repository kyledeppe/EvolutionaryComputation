import random
from evocomp import EvoComp
from forwardnet import ForwardNet
import matplotlib.pyplot as plt


def binary_wine():
    """
    Uploads crisp data, fuzzifies it, and implements ec to try and find the best network
    :return: None
    """
    training_file = 'cross_data (3 inputs - 2 outputs).csv'  # 3 inputs, 2 outputs

    num_inputs = 3
    num_outputs = 2

    training_data = ForwardNet.load_training_data(training_file, num_inputs)
    random.shuffle(training_data)  # Shuffle the training data to keep results useful
    num_samples = None  # Number of samples from the training data to use (reduces amount of computation)
    population_size = 50  # Number of parents to start out with
    selection_size = 50  # Number of survivors between the parents and the offspring (mu + lambda)
    generations = 15  # Iterations

    min_weight = -1  # Min value for weights and biases
    max_weight = 1  # Max value for weights and biases
    min_hidden = 5  # Min number of hidden nodes in a layer
    max_hidden = 10  # Max number of hidden nodes in a layer
    min_layer = 1  # Min number of layers
    max_layer = 3  # Max number of layers
    mut_chance = 50  # Mutation chance for each individual weight

    ec = EvoComp(init_population_size=population_size, selection_size=selection_size, generations=generations,
                 training_data=training_data, num_samples=num_samples, num_inputs=num_inputs, num_outputs=num_outputs,
                 min_weight=min_weight, max_weight=max_weight, min_hidden=min_hidden, max_hidden=max_hidden,
                 min_layer=min_layer, max_layer=max_layer, input_type='raw_input', output_type='raw_output',
                 type_init='select', mut_chance=mut_chance, mutate_data=False)

    ec.evolutionary_computation()

    coordinate_x = []
    coordinate_y = []
    colors = []
    precision = 0.001  # Concentration of dots on graph
    multiplier = int(1 / (precision * 100))
    for i in range(42 * multiplier + 1):  # -2.1 to +2.1 on graph
        for j in range(42 * multiplier + 1):  # -2.1 to +2.1 on graph
            num1 = (float(i) / (10 * multiplier)) - 2.1
            num2 = (float(j) / (10 * multiplier)) - 2.1
            coordinate_x.append(num1)
            coordinate_y.append(num2)
            coordinate = [num1, num2, 0]  # 3rd coordinate is 0 because it doesn't alter output
            network_input = {'raw_input': coordinate}  # How ForwardNet takes an input into neural network
            outputs = ec.best_nn.forward_computation(network_input)  # Outputs[-1] is output layer's outputs
            colors.append('r' if outputs[-1].index(max(outputs[-1])) == 0 else 'b')  # Red if class 1, else blue

    plt.scatter(coordinate_x, coordinate_y, s=10, c=colors, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    binary_wine()
