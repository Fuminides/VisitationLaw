'''
This script checks the mobility law for the US


'''

import utils
import geoutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

states = geoutils.get_state_names()
distances = geoutils.state_distance()

def incoming_pop_distance(state):
    # Sort states by distance
    sorted_states = distances[state].sort_values(ascending=False).index
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue

        pop_flow = edges.loc[o_state, state]
        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result

def outgoing_pop_distance(state):
    # Sort states by distance
    sorted_states = distances.loc[state].sort_values(ascending=False).index
    result = pd.DataFrame(np.zeros((len(states), 3)), index=sorted_states, columns=['distance', 'pop_flow', 'pop_flow_distance'])
    
    for o_state in sorted_states:
        if o_state == state:
            continue

        pop_flow = edges.loc[state, o_state]
        distance = distances.loc[o_state, state]
        result.loc[o_state, 'distance'] = distance
        result.loc[o_state, 'pop_flow'] = pop_flow
        result.loc[o_state, 'pop_flow_distance'] = pop_flow / distance

    result.sort_values(by='distance', ascending=False, inplace=True)
    return result

import sys

try:
    state = sys.argv[1]
    data_folder_path = sys.argv[2]
except:
    state = 'California'
    data_folder_path = './MyStuff/Data/'

edges = geoutils.load_travel_data(data_folder_path, verbose=True)
california_incoming = incoming_pop_distance(state)
california_outgoing = outgoing_pop_distance(state)
plt.plot(california_incoming['distance'], california_incoming['pop_flow'], 'r')
plt.plot(california_outgoing['distance'], california_outgoing['pop_flow'], 'b')
plt.legend(['Incoming', 'Outgoing'])
plt.title('Population flow vs distance for {}'.format(state))
plt.xlabel('Distance (km)')
plt.ylabel('Population flow')
plt.show()

print('Done')