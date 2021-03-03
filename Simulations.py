from environment import Simulation
import pygame
from time import sleep
import csv
import numpy as np

# This will be your subscriber node
# You will need to create a callback function
# You will need to move sim.move(command) inside the callback
# command can only be one of the following "south", "north", "west", "east".
# Remember to initiate a ROS node and subscribe to the topic /cmd

def simulate_Markovian(policy_file, config_file, time_horizon, row_number, column_number):

    policy = {}
    print(policy_file)
    with open(policy_file,'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k in csv_reader:
            policy[k[0]] = k[1]
    sim = Simulation(config_file)
    num_of_states = row_number * column_number
    policy_modified = [{} for i in range(int(time_horizon))]
    for key in policy.keys():
        time_step = int(int(key)/int(num_of_states))
        row = int(row_number-1- np.floor((int(key) % int(num_of_states))/column_number))
        column = (int(key) % int(num_of_states))%column_number
        if policy[key] == 'S':
            policy_modified[time_step][(column, row)] = 'south'
        elif policy[key] == 'E':
            policy_modified[time_step][(column, row)] = 'east'
        elif policy[key] == 'N':
            policy_modified[time_step][(column, row)] = 'north'
        elif policy[key] == 'W':
            policy_modified[time_step][(column, row)] = 'west'
        elif policy[key] == 'L':
            policy_modified[time_step][(column, row)] = 'stay'
    print(policy_modified)
    state = sim.get_state() #grab state
    done = False
    counter = 0
    while not done:
        action = policy_modified[counter][state['agents'][0]]
        print(action)
        done, state = sim.move(action) #main call
        if counter == 0:
            sleep(5)
        sleep(0.5)
        counter = counter +1

def simulate_Stationary(policy_file, config_file, row_number, column_number):

    policy = {}
    with open(policy_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            policy[row[0]] = row[1]
    sim = Simulation(config_file)
    policy_modified = {}
    for key in policy.keys():
        row = row_number-1- int(np.floor(int(key)/column_number))
        column = int(key)%column_number
        if policy[key] == 'S':
            policy_modified[(column, row)] = 'south'
        elif policy[key] == 'E':
            policy_modified[(column, row)] = 'east'
        elif policy[key] == 'N':
            policy_modified[(column, row)] = 'north'
        elif policy[key] == 'W':
            policy_modified[(column, row)] = 'west'
        elif policy[key] == 'L':
            policy_modified[(column, row)] = 'stay'

    state = sim.get_state() #grab state
    done = False
    counter = 0
    while not done:
        action = policy_modified[state['agents'][0]]
        print(action)
        done, state = sim.move(action) #main call
        if counter == 0:
            sleep(5)
        sleep(0.5)
        counter = counter +1
