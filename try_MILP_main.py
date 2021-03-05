import numpy as np
from grid_world import grid_world
from random import randint,seed
from MDP_class import MDP
from graph import Graph
# from Simulations import simulate_Markovian, simulate_Stationary
# import csv

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Environment Specifications'''

# Define the grid world environment
row, column = 8,8
goals = [63]
absorb = goals
init = 0
slip = 0
model = grid_world(row,column,absorb,slip)
num_of_states = len(model[0])

# Define the discount factor
discount = 0.9


base_MDP = MDP(model)
costs= {}
for state in base_MDP.states():
    for act in base_MDP.active_actions()[state]:
        if state not in absorb:
            costs[(state,act)] = np.random.randint(5)
        else:
            costs[(state,act)] = 0

policy = base_MDP.MILP_min_discounted_cost_max_reach(init, goals, absorb, costs, discount)
print(policy)
