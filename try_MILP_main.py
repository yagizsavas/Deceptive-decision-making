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
row, column = 10,10
goals = [90]
absorb = goals
init = 0
slip = 0.1
model = grid_world(row,column,absorb,slip)
num_of_states = len(model[0])

# Define the discount factor
discount = 0.9


base_MDP = MDP(model)

max_reach_val,reach_policy = base_MDP.compute_discounted_max_reach(init, goals, discount = 0.2)
reach_reward = base_MDP.reward_for_reach(goals)
reach_probability, _ = base_MDP.value_evaluate(reach_reward ,reach_policy, discount = 0.2)
print(reach_probability[init])

costs= {}
for state in base_MDP.states():
    for act in base_MDP.active_actions()[state]:
        if state not in absorb:
            costs[(state,act)] = np.random.randint(5)
        else:
            costs[(state,act)] = 0

min_cost_val,policy = base_MDP.compute_min_cost_subject_to_max_reach(init,goals,absorb, costs, discount = 0.8)

#policy = base_MDP.MILP_min_discounted_cost_max_reach(init, goals, absorb, costs, discount)
print(policy)
