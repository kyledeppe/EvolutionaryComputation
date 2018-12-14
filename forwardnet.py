"""
This class provides the forward computation for the evolutionary computation and backpropagation algorithms. In
addition, it handles loading data, fuzzification, error calculation, weight mutation, testing data, and displaying
results.
"""
import numpy as np
import random
from csv import reader
from scipy.special import expit


class ForwardNet:
    ACTIVATION = {'sigmoid': lambda x: float(expit(x)),
                  'tanh': lambda x: float(np.tanh(x)),
                  'relu': lambda x: x if x >= 0 else 0,
                  'identity': lambda x: x,
                  'sinusoid': lambda x: np.sin(x)}

    def __init__(self, training_data=None, num_samples=None, layer_counts=None, min_weight=-1, max_weight=1,
                 weights=None, biases=None, weight_files=None, bias_files=None, weight_init='random', input_type=None,
                 output_type=None, type_init=None, function=None, mut_chance=20, mutate_data=True):
        """
        This class is used for the forward computation section of a neural net, in addition to general operations
        :param training_data: A list of dictionaries of tuples -> [row]{'raw_input'}[coordinate]
        :param num_samples: Number of samples actually used from training data
        :param layer_counts: List of every layer's node count, starting from input layer
        :param min_weight: Min value of weights and biases
        :param max_weight: Max value of weights and biases
        :param weights: List of list of weights -> [layer][node][weight]
        :param biases: List of list of biases -> [layer][node][bias]
        :param weight_files: Optionally add weights via a csv file, similarly to Lab 1A.
        :param bias_files: Optionally add biases via a csv file, similarly to Lab 1A.
        :param weight_init: The method of how weights and biases are determined
        :param input_type: 'raw_input', or a type of fuzzy input
        :param output_type: 'raw_output', or a type of fuzzy output
        :param type_init: The method of how input and output types are determined
        :param function: Activation function's name according to the ACTIVATION dictionary
        :param mut_chance: The mutation chance for each individual weight
        :param mutate_data: Option to mutate data from raw to fuzzy
        """
        if num_samples is None:
            self.training_data = training_data
            self.testing_data = self.training_data[:]
        else:
            self.testing_data = training_data[:num_samples]
            self.training_data = training_data[:num_samples]

        self.layer_counts = layer_counts
        self.num_layers = len(self.layer_counts)

        if weight_init == 'random':
            self.current_biases = [self.init_weights_random(rows=i, columns=1, min_val=min_weight, max_val=max_weight)
                                   for i in self.layer_counts[1:]]
            self.current_weights = [self.init_weights_random(rows=self.layer_counts[i + 1],
                                                             columns=self.layer_counts[i],
                                                             min_val=min_weight, max_val=max_weight)
                                    for i in range(len(self.layer_counts) - 1)]
        elif weight_init == 'select':
            self.current_biases = biases
            self.current_weights = weights
        elif weight_init == 'files':
            self.current_biases = [self.load_csv_file(i) for i in bias_files]
            self.current_weights = [self.load_csv_file(i) for i in weight_files]
        else:
            raise TypeError('weight_init can only be "random", "select", or "files"')

        self.type_init = type_init
        if self.type_init == 'random':
            self.input_type = self.choose_fuzzy_type('in')
            self.output_type = self.choose_fuzzy_type('out')
        elif self.type_init == 'select':
            self.input_type = input_type
            self.output_type = output_type
        else:
            raise TypeError('type_init can only be "random" or "select"')

        if function is None:
            self.function = self.choose_activation_function()
        else:
            self.function = function

        self.mut_chance = mut_chance

        self.prev_sum_errors = 0
        self.current_sum_errors = 0
        self.change_in_error = self.prev_sum_errors - self.current_sum_errors
        self.all_sum_errors = []

        self.percent_correct = None
        self.iterations = 0
        self.score = 0
        self.prob_selected = None

        self.error_matrix = None

        self.mutate_data = mutate_data
        self.multipliers = [-2, -1, -0.3, 0.3, 2]

    def forward_computation(self, sample):
        """
        Output = ActivationFunction(Weights * Inputs + Bias)
        :param sample: A row from the training data
        :return: List of outputs from every layer
        """
        outputs = []
        output = sample[self.input_type]

        for i in range(self.num_layers - 1):
            output = [self.ACTIVATION[self.function](
                np.dot(output, self.current_weights[i][neuron]) + self.current_biases[i][neuron][0])
                for neuron in range(self.layer_counts[i + 1])]
            outputs.append(output)

        return outputs

    def test_network(self):
        """
        Measures the percentage of experiment data outputs that match their expected value
        :return: None
        """
        correct, wrong = 0, 0
        errors = []

        if self.percent_correct is None:
            self.error_matrix = [[0] * self.layer_counts[-1] for _ in range(self.layer_counts[-1])]
            for sample in self.testing_data:
                outputs = self.forward_computation(sample)
                error = self.find_error(outputs[-1], sample)
                errors.append(error)

                expected_value_index = sample[self.output_type].index(max(sample[self.output_type]))
                output_indexes = sorted(range(len(outputs[-1])), key=lambda i: outputs[-1][i], reverse=True)[:3]

                # fuzzy_range allows adjacent indexes to count as correct for fuzzificaton purposes
                fuzzy_range = 0
                for output_index in output_indexes:
                    if outputs[-1][output_index] == max(outputs[-1]):
                        if (output_index - expected_value_index) % len(sample[self.output_type]) <= fuzzy_range:
                            correct += 1
                        else:
                            wrong += 1
                        self.error_matrix[expected_value_index][output_index] += 1

            self.current_sum_errors = self.calc_mean_squared_errors(errors)
            self.percent_correct = 100 * correct / (correct + wrong)

            self.score = self.percent_correct

        print('percent correct: {:>3}%{:11}sum errors: {:<20}'.format(
            format(self.percent_correct, '.0f'), '',
            format(self.current_sum_errors, '.4f')))

    def find_error(self, output, sample):
        """
        Finds the individual error for every output neuron and returns a list
        :param output: List of outputs of every output neuron
        :param sample: Row from training data
        :return: List of errors
        """
        error = [sample[self.output_type][neuron] - output[neuron] for neuron in range(self.layer_counts[-1])]
        return error

    def calc_mean_squared_errors(self, errors):
        """
        Finds the mean sum of squared errors for a list of errors
        :param errors: The errors from one epoch
        :return: The mean sum of squared errors
        """
        def mse(error):
            sample_size = len(self.training_data)
            return error ** 2 / (2 * sample_size)

        mean_sum_squared_errors = sum([mse(neuron) for error in errors for neuron in error])
        return mean_sum_squared_errors

    def print_results(self, generations=None):
        """
        Prints the weights, biases, activation function, input/output types, layer counts, epochs,
        percentage correct, and MSE of the neural network
        :return: None
        """
        print('\n' * 2)
        for layer in range(len(self.current_biases)):
            print('Biases - Layer {}'.format(layer + 1))
            print('{:<20}'.format('Node') + '{:<19}'.format('Bias'))
            for node in range(len(self.current_biases[layer])):
                print('b{:<5}'.format(node + 1), end='')
                for bias in range(len(self.current_biases[layer][node])):
                    print('{:>20}'.format(format(self.current_biases[layer][node][bias], '.4f')), end='')
                print()
            print()

        print('\n')
        for layer in range(len(self.current_weights)):
            print('Weights - Layer {}'.format(layer + 1))
            print('{:<20}'.format(''), end='')
            for weight in range(len(self.current_weights[layer][0])):
                print('x{:<19}'.format(weight + 1), end='')
            print()
            for node in range(len(self.current_weights[layer])):
                print('w{:<5}'.format(node + 1), end='')
                for weight in range(len(self.current_weights[layer][node])):
                    print('{:>20}'.format(format(self.current_weights[layer][node][weight], '.4f')), end='')
                print()
            print()

        print('\n')
        print('NN Info')
        print('Activation Function: {}'.format(self.function))
        print('Input Type: {}'.format(self.input_type))
        print('Output Type: {}'.format(self.output_type))
        print('Layers: {}'.format(self.layer_counts))
        print('Multipliers: {}'.format(self.multipliers))
        if generations is not None:
            print('Number of Generations: {}'.format(generations, 4))
        if self.iterations != 0:
            print('Epochs of Training: {}'.format(self.iterations))
        print('Percent Correct: {}'.format(self.percent_correct))
        print('Mean Sum of Squared Errors: {}'.format(round(self.current_sum_errors, 4)))
        if self.score != 0:
            print('Score: {}'.format(round(self.score, 4)))
        print('Error Matrix: {}'.format(self.error_matrix))
        print('\n')

    def mutate_network(self):
        """
        Tries to mutate the weights, input/output types, and activation function
        :return: None
        """
        def try_mutate(weight):
            if random.randint(1, 100) <= self.mut_chance:
                weight *= random.choice(self.multipliers)

            return weight

        self.current_weights = [[[try_mutate(weight)
                                  for weight in neuron]
                                 for neuron in layer]
                                for layer in self.current_weights]

        self.current_biases = [[[try_mutate(bias)
                                 for bias in neuron]
                                for neuron in layer]
                               for layer in self.current_biases]

        if self.mutate_data:
            if random.randint(1, 100) <= self.mut_chance:
                self.input_type = self.choose_fuzzy_type('in')
            if random.randint(1, 100) <= self.mut_chance:
                self.output_type = self.choose_fuzzy_type('out')

        if random.randint(1, 100) <= self.mut_chance:
            self.function = self.choose_activation_function()

    @staticmethod
    def init_weights_zero(data):
        """
        Takes a table and returns an equivalently sized table of zeros
        :param data: The table that will have its size copied
        :return: Table of zeros
        """
        row_size = len(data[0])
        table_zeros = [[0] * row_size for _ in data]

        return table_zeros

    @staticmethod
    def init_weights_random(rows, columns, min_val=-1, max_val=1):
        """
        Creates a table of size rows x columns
        :param rows: Number of rows
        :param columns: Number of columns
        :param min_val: Min weight value
        :param max_val: Max weight value
        :return: A table of random numbers
        """
        min_value = min_val
        max_value = max_val
        table_randoms = [[random.uniform(min_value, max_value) for _ in range(columns)] for _ in range(rows)]

        return table_randoms

    @staticmethod
    def init_hidden_layers(input_size, output_size, min_hidden=5, max_hidden=15, min_layer=2, max_layer=5):
        """
        Creates a list of the number of nodes in every layer
        :param input_size: Number of nodes in input layer
        :param output_size: Number of nodes in output layer
        :param min_hidden: Min hidden nodes
        :param max_hidden: Max hidden nodes
        :param min_layer: Min nodes per layer
        :param max_layer: Max nodes per layer
        :return: List of nodes in every layer
        """
        min_nodes = min_hidden  # Minimum hidden nodes per layer
        max_nodes = max_hidden  # Maximum hidden nodes per layer
        min_layers = min_layer  # Minimum hidden layers
        max_layers = max_layer  # Maximum hidden layers

        return [input_size] + \
               [random.randint(min_nodes, max_nodes) for _ in range(random.randint(min_layers, max_layers))] + \
               [output_size]

    @staticmethod
    def fuzzy_triangle(sample_input):
        """
        Takes a crisp vector and returns a fuzzy vector
        :param sample_input: Crisp input or output
        :return: Vector with fuzzy triangles
        """
        size = len(sample_input)
        triangle_input = [float(0)] * size

        for i, value in enumerate(sample_input):
            value_f = float(value)
            triangle_input[i - 2] += value_f * 0.25
            triangle_input[i - 1] += value_f * 0.5
            triangle_input[i] += value_f
            triangle_input[(i + 1) % size] += value_f * 0.5
            triangle_input[(i + 2) % size] += value_f * 0.25

        return triangle_input

    @staticmethod
    def fuzzy_trapezoid(sample_input):
        """
        Takes a crisp vector and returns a fuzzy vector
        :param sample_input: Crisp input or output
        :return: Vector with fuzzy trapezoids
        """
        size = len(sample_input)
        trapezoid_input = [float(0)] * size

        for i, value in enumerate(sample_input):
            value_f = float(value)
            trapezoid_input[i - 3] += value_f * 0.25
            trapezoid_input[i - 2] += value_f * 0.75
            trapezoid_input[i - 1] += value_f
            trapezoid_input[i] += value_f * 1.1  # 1.1 instead of 1 so that it's possible to tell "true" outputs
            trapezoid_input[(i + 1) % size] += value_f
            trapezoid_input[(i + 2) % size] += value_f * 0.75
            trapezoid_input[(i + 3) % size] += value_f * 0.25

        return trapezoid_input

    @staticmethod
    def choose_fuzzy_type(type_io):
        """
        Randomly returns a type of input
        :param type_io: 'in' or 'out'
        :return: A crisp or fuzzy i/o type
        """
        if type_io == 'in':
            options = ('raw_input', 'triangle_input', 'trapezoid_input')
            return options[random.randint(0, len(options) - 1)]
        elif type_io == 'out':
            options = ('raw_output', 'triangle_output', 'trapezoid_output')
            return options[random.randint(0, len(options) - 1)]
        else:
            raise TypeError('choose_fuzzy_type() only accepts "in" or "out" as parameters.')

    @staticmethod
    def choose_activation_function():
        """
        Returns a random activation function
        :return: String of chosen activation function
        """
        options = ('sigmoid', 'tanh', 'relu', 'identity', 'sinusoid')
        return options[random.randint(0, len(options) - 1)]

    @staticmethod
    def load_and_fuzzify_training_data(csv_file, num_inputs):
        """
        Returns data set with crisp inputs/outputs in addition to their fuzzy forms
        :param csv_file: A csv file with inputs on the left and outputs on the right, no spaces
        :param num_inputs: The number of inputs the file contains
        :return: A list of dictionaries of tuples, where tuples hold coordinates and dictionaries hold the type of i/o
        """
        with open(csv_file, 'r') as csv_file_ptr:
            csv_reader = reader(csv_file_ptr, delimiter=',')
            table = [{'raw_input': tuple(float(value) for value in row[:num_inputs]),
                      'raw_output': tuple(float(value) for value in row[num_inputs:]),
                      'triangle_input': ForwardNet.fuzzy_triangle(row[:num_inputs]),
                      'triangle_output': ForwardNet.fuzzy_triangle(row[num_inputs:]),
                      'trapezoid_input': ForwardNet.fuzzy_trapezoid(row[:num_inputs]),
                      'trapezoid_output': ForwardNet.fuzzy_trapezoid(row[num_inputs:])} for row in csv_reader]

            return table

    @staticmethod
    def load_training_data(csv_file, num_inputs):
        """
        Used for datasets that don't need to be fuzzified
        :param csv_file: A .csv file
        :param num_inputs: The number of samples in the training set
        :return: Table of floats
        """
        with open(csv_file, 'r') as csv_file_ptr:
            csv_reader = reader(csv_file_ptr, delimiter=',')
            table = [{'raw_input': tuple(float(value) for value in row[:num_inputs]),
                      'raw_output': tuple(float(value) for value in row[num_inputs:])} for row in csv_reader]

            return table

    @staticmethod
    def load_csv_file(csv_file):
        """
        Used for Lab 1A, loads weights and biases from files as floats
        :param csv_file: A .csv file
        :return: Table of floats
        """
        with open(csv_file, 'r') as csv_file_ptr:
            csv_reader = reader(csv_file_ptr, delimiter=',')
            table = [tuple(float(value) for value in row) for row in csv_reader]

        return table
