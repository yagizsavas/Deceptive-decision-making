import numpy as np
from grid_world import grid_world
from random import randint,seed
from MDP_class import MDP
from graph import Graph
import csv

row, column = 17,17
goals = [288,16,6,176]
absorb = set([5, 5+17, 5+ 3*17, 5+ 4*17, 5 + 5*17, 5 + 6*17, 5 + 7*17])
absorb.update(set([5+ 9*17, 5+ 10*17, 5 + 11*17, 5 + 12*17, 5 + 13*17, 5 + 15*17, 5 + 16*17]))
absorb.update( set([11, 11+17, 11+ 3*17, 11+ 4*17, 11 + 5*17, 11 + 6*17, 11 + 7*17]))
absorb.update(set([11+ 9*17, 11+ 10*17, 11 + 11*17, 11 + 12*17, 11 + 13*17, 11 + 15*17, 11 + 16*17]))
absorb.update(set([85,86,88,89,90,91,92, 94,95,96,97,98, 100, 101]))
absorb.update(set([187,188,190,191,192,193,194,196,197,198,199,200,202,203]))
absorb.update(set(goals))
print(list(sorted(absorb)))
init = 272
slip = 0
model = grid_world(row,column,absorb,slip)
MDP = MDP(model)


true_goal = goals[0]
prior_goals = [0.5,0.5,0.5,0.5]
discount = 0.9

graph = Graph(model)
min_times = graph.Floyd_Warshall()

rewards=[{} for k in range(len(goals))]
for k in range(len(goals)):
    for state in MDP.states():
        for act in MDP.active_actions()[state]:
            if state != goals[k]:
                rewards[k][(state,act)] = -60
            else:
                rewards[k][(state,act)] = 0

goal_values = []
for k in range(len(goals)):
    goal_values.append( MDP.soft_max_val_iter( rewards[k] , [goals[k]] , discount ) )

goal_posteriors=[{} for k in range(len(goals))]
for state in MDP.states():
    for act in MDP.active_actions()[state]:
        for k in range(len(goals)):
            goal_posteriors[k][(state,act)] = np.exp(goal_values[k][state]-goal_values[k][init])*prior_goals[k]
        denum=np.array([goal_posteriors[i][(state,act)] for i in range(len(goals))])
        for k in range(len(goals)):
            goal_posteriors[k][(state,act)] = goal_posteriors[k][(state,act)]/ np.sum(denum)
print(goal_posteriors)
MDP_costs= {}
for state in MDP.states():
    for act in MDP.active_actions()[state]:
        if state not in absorb:
            most_probable_decoy = np.amax([goal_posteriors[i][(state,act)] for i in range(1,len(goals))])
            deception_index =  most_probable_decoy - goal_posteriors[0][(state,act)]
            #deception_index = most_probable_decoy
            MDP_costs[(state,act)] = (1- deception_index)
        else:
            MDP_costs[(state,act)] = 0
print(MDP_costs)

counter = 0
while init != true_goal:
    [a,policy] = MDP.compute_min_cost_subject_to_max_reach(init,[true_goal], absorb, MDP_costs)
    counter = counter +1
    init = model[1][(init,policy[init])][1][0]
    for k in MDP_costs.keys():
        MDP_costs[k]  = MDP_costs[k]* discount** min_times[(init,k[0])]
    print(policy)
print(policy)
w = csv.writer(open("gridsim-master/output.csv", "w"))
for key, val in policy.items():
    w.writerow([key, val])
