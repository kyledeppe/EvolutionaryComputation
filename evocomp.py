"""
This class provides the evolutionary computation for the creation of neural nets. It is used in conjunction with the
ForwardNet class.
"""
import copy
import random
from forwardnet import ForwardNet


class EvoComp:
    def __init__(self, init_population_size=100, selection_size=100, generations=10, training_data=None, num_samples=100,
                 num_inputs=23, num_outputs=13, min_weight=-1, max_weight=1, min_hidden=5, max_hidden=15, min_layer=2,
                 max_layer=5, input_type=None, output_type=None, type_init='random', mut_chance=20, mutate_data=True):
        """
        This class implements the evolutionary computation algorithm with ForwardNet implementing the
        forward computation that tests the generated neural nets.
        :param init_population_size: Number of initial parents
        :param selection_size: Amount of selected individuals in (mu + lambda) configuration, i.e. out of the
        pool of parents and offspring.
        :param generations: Iterations of the algorithm
        :param mutate_data: Option to mutate data from raw to fuzzy
        """
        self.init_population_size = init_population_size
        self.selection_size = selection_size
        self.generations = generations

        self.neural_nets = self.init_population(training_data=training_data, num_inputs=num_inputs,
                                                num_outputs=num_outputs, num_samples=num_samples,
                                                min_weight=min_weight, max_weight=max_weight, min_hidden=min_hidden,
                                                max_hidden=max_hidden, min_layer=min_layer,
                                                max_layer=max_layer, input_type=input_type, output_type=output_type,
                                                type_init=type_init, mut_chance=mut_chance, mutate_data=mutate_data)

        self.best_score = float('-inf')
        self.best_nn = None

    def evolutionary_computation(self):
        """
        Iterates through specified number of generations and displays the best neural network
        :return: None
        """
        for i in range(self.generations):
            print('\nGeneration {}\n'.format(i + 1))
            self.neural_nets = self.create_offspring(self.neural_nets[:])

            for nn in self.neural_nets:
                nn.test_network()

                if nn.score > self.best_score:
                    self.best_score = nn.score
                    self.best_nn = copy.deepcopy(nn)

            self.neural_nets = self.sus(self.selection_size)

        self.best_nn.print_results(self.generations)

    def init_population(self, training_data, num_inputs, num_outputs, num_samples, min_weight=-1, max_weight=1,
                        min_hidden=5, max_hidden=15, min_layer=2, max_layer=5, input_type=None, output_type=None,
                        type_init='random', mut_chance=20, mutate_data=True):
        """
        Returns a list of neural networks with randomized features
        :param training_data: Inputs and outputs of the training data
        :param num_inputs: Number of inputs in the neural network
        :param num_outputs: Number of outputs in the neural network
        :param num_samples: Number of samples from the training data to use (reduces amount of computation)
        :param min_weight: Min weight that can be initialized
        :param max_weight: Max weight that can be initialized
        :param min_hidden: Min number of hidden nodes in a given layer
        :param max_hidden: Max number of hidden nodes in a given layer
        :param min_layer: Min number of hidden layers
        :param max_layer: Max number of hidden layers
        :param input_type: 'raw_input', or a type of fuzzy input
        :param output_type: 'raw_output', or a type of fuzzy output
        :param type_init: 'select' or 'random'
        :param mut_chance: Mutation chance for each individual weight
        :param mutate_data: Option to mutate data from raw to fuzzy
        :return: List of neural networks
        """
        return [ForwardNet(training_data=training_data, weight_init='random',
                           layer_counts=ForwardNet.init_hidden_layers(num_inputs, num_outputs, min_hidden=min_hidden,
                                                                      max_hidden=max_hidden, min_layer=min_layer,
                                                                      max_layer=max_layer), min_weight=min_weight,
                           max_weight=max_weight, num_samples=num_samples, input_type=input_type,
                           output_type=output_type, type_init=type_init, mut_chance=mut_chance, mutate_data=mutate_data)
                for _ in range(self.init_population_size)]

    @staticmethod
    def create_offspring(parents):
        """
        Creates the next generation
        :param parents: The selected survivors of the previous generation
        :return: The new generation via the u+lambda configuration
        """
        offspring = []

        # Check for even or odd number of parents
        if len(parents) % 2 == 0:
            for i in range(0, len(parents), 2):
                offspring = offspring + EvoComp.one_point_crossover(parents[i], parents[i + 1])
        else:
            for i in range(0, len(parents)-1, 2):
                offspring = offspring + EvoComp.one_point_crossover(parents[i], parents[i + 1])

        for child in offspring:
            child.mutate_network()

        return parents + offspring

    @staticmethod
    def one_point_crossover(parent1, parent2):
        """
        This algorithm performs a structural one point crossover on two parents and returns two offspring. First, it
        randomly assigns each offspring one of two activation functions, input types, and output types. Next, it
        chooses what layers both parents will be split at. Finally, the layer counts, weights, and biases are switched
        accordingly and assigned to the children. The parents stay the same during this process to enable mu+lambda
        selection.
        :type parent1: NeuralNetwork
        :type parent2: NeuralNetwork
        :param parent1: Parent creating an offspring
        :param parent2: Parent creating an offspring
        :return: The two resulting children of the switched halves
        """
        def random_choice(choice1, choice2):
            if random.choice([True, False]):
                return choice1, choice2
            else:
                return choice2, choice1

        def cross_weights(weights1, weights2, trade1, trade2, layer_counts, type_weight):
            # Switch the weights where they were cut off
            new_weights = weights1[:trade1] + weights2[trade2:]

            # Regenerate the weights or biases at the location where they were cut off to ensure they're the correct
            # size
            if type_weight == 'weight':
                new_weights[trade1 - 1] = ForwardNet.init_weights_random(layer_counts[trade1], layer_counts[trade1 - 1])
            elif type_weight == 'bias':
                new_weights[trade1 - 1] = ForwardNet.init_weights_random(layer_counts[trade1], 1)

            return new_weights

        # Assign the activation functions, input types, and output types
        child1_function, child2_function = random_choice(parent1.function, parent2.function)
        child1_input, child2_input = random_choice(parent1.input_type, parent2.input_type)
        child1_output, child2_output = random_choice(parent1.output_type, parent2.output_type)

        # Randomly choose which hidden layer they will be cut off at
        num_trade1 = random.randint(1, parent1.num_layers - 2)
        num_trade2 = random.randint(1, parent2.num_layers - 2)
        child1_layer_counts = parent1.layer_counts[:num_trade1] + parent2.layer_counts[num_trade2:]
        child2_layer_counts = parent2.layer_counts[:num_trade2] + parent1.layer_counts[num_trade1:]

        # Switch the weights according to the layers that were switched and generate the weights that are spliced during
        # the switching process.
        child1_weights = cross_weights(parent1.current_weights, parent2.current_weights, num_trade1, num_trade2,
                                       child1_layer_counts, 'weight')
        child2_weights = cross_weights(parent2.current_weights, parent1.current_weights, num_trade2, num_trade1,
                                       child2_layer_counts, 'weight')
        child1_biases = cross_weights(parent1.current_biases, parent2.current_biases, num_trade1, num_trade2,
                                      child1_layer_counts, 'bias')
        child2_biases = cross_weights(parent2.current_biases, parent1.current_biases, num_trade2, num_trade1,
                                      child2_layer_counts, 'bias')

        # The miracle of life
        child1 = ForwardNet(training_data=parent1.training_data, weights=child1_weights, biases=child1_biases,
                            weight_init='select', layer_counts=child1_layer_counts, input_type=child1_input,
                            output_type=child1_output, type_init='select', function=child1_function,
                            mutate_data=parent1.mutate_data)

        child2 = ForwardNet(training_data=parent2.training_data, weights=child2_weights, biases=child2_biases,
                            weight_init='select', layer_counts=child2_layer_counts, input_type=child2_input,
                            output_type=child2_output, type_init='select', function=child2_function,
                            mutate_data=parent2.mutate_data)

        return [child1, child2]

    def sus(self, num_kept):
        """
        Stochastic Universal Sampling selection method. This method creates an evenly distributed set of points with a
        random offset that get plugged in to a roulette wheel.
        :param num_kept: Number of parents that will be in the next generation
        :return: List of neural networks
        """
        sum_scores = sum(nn.score for nn in self.neural_nets)
        segment = sum_scores / num_kept
        offset = random.uniform(0, segment)
        offset_points = [i*segment + offset for i in range(num_kept)]

        return self.rws(offset_points)

    def rws(self, points):
        """
        Roulette Wheel Selection. This roulette wheel is a contiguous line of the fitness scores, where the chance a
        neural network gets chosen is calculated by fitness/(sum(fitnesses of population). When a point generated by
        the self.sus method lies within the respective range of the fitness of a neural network, the network is chosen
        to be part of the next generation.
        :param points: The points that will be plotted against the roulette wheel
        :return: List of neural networks
        """
        kept_population = []
        fitness_sum = self.neural_nets[0].score

        for point in points:
            i = 0
            while fitness_sum < point:
                i += 1
                fitness_sum += self.neural_nets[i].score
            kept_population.append(self.neural_nets[i])

        return kept_population
