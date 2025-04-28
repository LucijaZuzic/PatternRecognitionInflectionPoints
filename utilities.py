
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

cm = 1 / 2.54  # centimeters in inches

# Figure settings

def prepare_figure():
    plt.rcParams["svg.fonttype"] = "none"
    rc('font',**{'family':'Arial'})
    
    SMALL_SIZE = 5
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 5

    plt.rc('font', size = SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize = SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize = MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize = SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title
    plt.figure(figsize = (29.7 / 4 * cm, 29.7/ 5 * cm), dpi = 300)

# A preprocessing function for longitude and latitude values of a trajectory

def preprocess_longitude_latitude(longitude, latitude):
    x_return = [x - min(longitude) for x in longitude]
    y_return = [y - min(latitude) for y in latitude]
    if longitude[0] >= longitude[1]:
        x_return = [max(x_return) - x for x in x_return]
    if latitude[0] >= latitude[1]:
        y_return = [max(y_return) - y for y in y_return]
    return x_return, y_return

# A function defining the default probability of an unseen longitude or latitude offset

def dd():
    return 10 ** -20

# A function loading objects using pickle

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data

# A function generating longitude or latitude with predefined probability distributions depending on none, one, or two previous values

def predict_probability(points, probability, probability_in_next_step, probability_in_next_next_step):
    x = []
    previous_distance = 0
    previous_previous_distance = 0
    for i in range(points):
        if i == 0:
            distance = list(probability.keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability))],p=list(probability.values()))]
        else:
            if i == 1:
                if previous_distance in probability_in_next_step:
                    distance = list(probability_in_next_step[previous_distance].keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability_in_next_step[previous_distance]))],p=list(probability_in_next_step[previous_distance].values()))]
                else:
                    distance = list(probability.keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability))],p=list(probability.values()))]
            else:
                if previous_previous_distance in probability_in_next_next_step and previous_distance in probability_in_next_next_step[previous_previous_distance]:
                    distance = list(probability_in_next_next_step[previous_previous_distance][previous_distance].keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability_in_next_next_step[previous_previous_distance][previous_distance]))],p=list(probability_in_next_next_step[previous_previous_distance][previous_distance].values()))]
                else:
                    if previous_distance in probability_in_next_step:
                        distance = list(probability_in_next_step[previous_distance].keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability_in_next_step[previous_distance]))],p=list(probability_in_next_step[previous_distance].values()))]
                    else:
                        distance = list(probability.keys())[np.random.choice([ix_of_probability for ix_of_probability in range(len(probability))],p=list(probability.values()))]
        if isinstance(distance, list):
            distance = tuple(distance)
        if isinstance(distance, float):
            distance = tuple([distance])
        previous_previous_distance = previous_distance
        previous_distance = distance
        x.append(distance[0])
    return x