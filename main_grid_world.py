import numpy as np
from grid_world import grid_world
from random import randint,seed
from MDP_class import MDP
from graph import Graph
from Simulations import simulate_Markovian, simulate_Stationary
import csv

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Environment Specifications'''

# Define the grid world environment
row, column = 7,7
goals = [42,45,48]
true_goal = goals[0] # NOTE: Always choose the first element of the goals array as the true goal
absorb = goals
init = 0
slip = 0
model = grid_world(row,column,absorb,slip)
num_of_states = len(model[0])

# Define the time_horizon
time_horizon = 18

# Define the discount factor
discount = 0.9

# Define the cost of taking a step in the MDP (used in the shortest path computation)
one_step_cost = 30

# Define the prior distribution on the goals
prior_goals = np.ones((len(goals),1))/len(goals)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Define the posterior goal probabilities for each state'''

# Perform the computations on the base MDP model
base_MDP = MDP(model)

goal_posteriors = base_MDP.compute_goal_posteriors(prior_goals, init, goals, one_step_cost, discount)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Approximation algorithm 1'''

# This approximation algorithm performs on the base MDP model.
# It defines the costs for each state-action pair as follows:
#   Let c(s) be the posterior goal probability of the state s.
#   Let T(s) be the minimum time steps to reach the state s from the initial state.
#   Let base_MDP_costs(s,a) be the cost for the state-action pair (s,a), which is used in the linear problem
# We have base_MDP_costs(s,a) = c(s) * discount ** T(s)

graph = Graph(model)
min_times = graph.Dijkstra(init)
base_MDP_costs= {}
for state in base_MDP.states():
    for act in base_MDP.active_actions()[state]:
        if state not in absorb:
            base_MDP_costs[(state,act)] = goal_posteriors[0][(state,act)] * discount**min_times[state]
        else:
            base_MDP_costs[(state,act)] = 0

[a,policy] = base_MDP.compute_min_cost_subject_to_max_reach(init,[true_goal], absorb, base_MDP_costs)
f = open("grid_world_policy_1.csv", "w")
w_1 = csv.writer(f)
for key, val in policy.items():
    w_1.writerow([key, val])
f.close()
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Approximation algorithm 2'''

# This approximation algorithm performs on the product MDP model.
# For time_horizon N, the state-space of the product MDP model is S x N.
# The costs for each state-action pair is defined as follows:
#   Let c(s) be the posterior goal probability of the state s.
#   Let t be the current time step.
#   Let product_MDP_costs(s,a) be the cost for the state-action pair (s,a), which is used in the linear problem
# We have product_MDP_costs(s,a) = c(s) * discount ** t


product_model, absorb_product, true_goals_product = base_MDP.product_MDP(time_horizon, true_goal)
Product_MDP = MDP(product_model)

Product_MDP_costs= {}
for state in Product_MDP.states():
    for act in Product_MDP.active_actions()[state]:
        if state not in absorb_product:
            Product_MDP_costs[(state,act)] = goal_posteriors[0][(state % num_of_states , act)] * discount ** int(state/num_of_states)
        else:
            Product_MDP_costs[(state,act)] = 0

[a,policy] = Product_MDP.compute_min_cost_subject_to_max_reach(init,true_goals_product, absorb_product, Product_MDP_costs)
f = open("grid_world_policy_2.csv", "w")
w_2 = csv.writer(f)
for key, val in policy.items():
    w_2.writerow([key, val])
f.close()
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Simulations'''
config_file = "config_grid_world.txt"
f = open(config_file, 'w')
f.write('HEIGHT: '+str(row)+'\n')
f.write('WIDTH: '+str(column)+'\n')
f.write('BLOCK: agent '+str(int(init%column))+ ' '+str(int(row-1-np.floor(init/column)))+'\n')
for k in goals:
    f.write('BLOCK: goal '+str(int(k%column))+ ' '+str(int(row-1-np.floor(k/column)))+'\n')
f.close()

# Simulate the first policy
#policy_file = "grid_world_policy_1.csv"
#simulate_Stationary(policy_file, config_file, row, column)

print('testing fork')

# Simulate the second policy
policy_file = "grid_world_policy_2.csv"
simulate_Markovian(policy_file, config_file, time_horizon, row, column)
