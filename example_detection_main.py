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
init = 0
goals = [90]
obstacles = [21,22,23,31,32,33,63,64,65,73,74,75]
high_risk_states = [10,20,30,40,50,60,70,80]
mid_risk_states = list(range(41,44))+ list(range(51,54)) + list(range(61,63))\
                + list(range(71,73)) + list(range(81,84)) + list(range(91,94))
low_risk_states = list(range(1,10)) + list(range(11,20)) + list(range(24,30)) \
                +  list(range(34,40)) + list(range(44,50)) + list(range(54,60)) \
                + [66,67,68,69] + [76,77,78,79] + list(range(84,90)) + list(range(94,100))

absorb = goals + obstacles
slip = 0.1
beta = 0.9
model = grid_world(row,column,absorb,slip)
base_MDP = MDP(model)


graph = Graph(model)
min_times = graph.Dijkstra(init)

costs= {}
modified_costs = {}
for state in base_MDP.states():
    for act in base_MDP.active_actions()[state]:
        if state in high_risk_states:
            costs[(state,act)] = 4
            modified_costs[(state,act)] = beta**(min_times[state]-1)*4
        elif state in mid_risk_states:
            costs[(state,act)] = 2
            modified_costs[(state,act)] = beta**(min_times[state]-1)*2
        elif state in low_risk_states:
            costs[(state,act)] = 1
            modified_costs[(state,act)] = beta**(min_times[state]-1)*1
        else:
            costs[(state,act)] = 0
            modified_costs[(state,act)] = 0

_,constrained_MDP_policy = base_MDP.compute_min_cost_subject_to_max_reach(init,goals,absorb, costs, discount = beta)
print(constrained_MDP_policy)

_,undiscounted_policy = base_MDP.compute_min_cost_subject_to_max_reach(init,goals,absorb, costs, discount = 1)
print(undiscounted_policy)

_,proposed_policy = base_MDP.compute_min_cost_subject_to_max_reach(init,goals,absorb, modified_costs, discount = 1)
print(proposed_policy)

MILP_policy = base_MDP.MILP_min_discounted_cost_max_reach(init, goals, absorb, costs, beta)
print(MILP_policy)
