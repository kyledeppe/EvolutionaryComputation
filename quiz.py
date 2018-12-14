import random
from evocomp import EvoComp
from forwardnet import ForwardNet


def binary_wine():
    """
    Uploads crisp data, fuzzifies it, and implements ec to try and find the best network
    :return: None
    """
    training_file = 'binary_wine_inputs.csv'  # 22 inputs, 13 outputs

    num_inputs = 22
    num_outputs = 13

    training_data = ForwardNet.load_and_fuzzify_training_data(training_file, num_inputs)
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

    print('\n\n')
    preference = []
    print('Please take the following survey about your preferences in wine.')

    preference = preference + [0] * 2
    x = input('Do you like earthy flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    preference = preference + [0] * 5
    x = input('Do you like smoky flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    preference = preference + [0] * 1
    x = input('Do you like spicy flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    # preference = preference + [0] * 0
    x = input('Do you like nutty flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    preference = preference + [0] * 3
    x = input('Do you like floral flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    # preference = preference + [0] * 0
    x = input('Do you like berry flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    preference = preference + [0] * 1
    x = input('Do you like tropical flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    preference = preference + [0] * 2
    x = input('Do you like sour flavors? ')
    if 'y' in x or 'Y' in x:
        preference.append(1.0)
    else:
        preference.append(0.0)

    network_input = {'raw_input': preference}
    outputs = ec.best_nn.forward_computation(network_input)
    output_indexes = sorted(range(len(outputs[-1])), key=lambda i: outputs[-1][i], reverse=True)[:3]

    master_outputs = ["pinot noir", "merlot", "sangiovese", "tempranillo", "cabernet sauvignon", "syrah", "pinot grigio", "riesling", "sauvignon blanc", "moscato", "gewürztraminer", "viognier", "chardonnay"]

    print('\nYou might like:')
    for j in output_indexes:
        print(master_outputs[j])


if __name__ == '__main__':
    binary_wine()
