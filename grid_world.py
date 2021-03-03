##################################
# This code forms a grid world environment.
# Inputs:
# row: number of rows (integer)
# column: number of columns  (integer)
# absorb: list of absorbing states (list of integers)
# slip: slip probability
##################################
# Grid world structure for input (3,3,[8],slip):
#-------------------------
#|       |       |       |
#|   6   |   7   |  8 (a)|
#-------------------------
#|       |       |       |
#|   3   |   4   |   5   |
#-------------------------
#|       |       |       |
#|   0   |   1   |   2   |
#-------------------------

# =======================================

# Available actions are (N,W,S,E,L), i.e., counter-clockwise directions
# North, West, South, East, Loop

# Each action takes the agent to the desired successor state with 1-slip
# probability, and the agent stays in the same state with probability slip.

# The model is a list of two dictionaries. The first dictionary holds the list
# of available actions in a given state. For example, model[0][state] returns the list
# of available actions in that state, e.g., model[0][0]=['N','E','L'].
# The second dictionary holds the tuple of two lists for a given state-action pair.
# model[1][(state,action)] returns a tuple of two lists; first list is the
# transition probabilities and the second list is the successor states, e.g.,
# model[1][(0,'N')] = ([slip,1-slip] , [0,column]). In plain English, from state 0
# under the action 'N', the agent transitions to the state 'column' with
# probability 1-slip and to the state 0 with probability slip.

# =========================================

def grid_world(row,column,absorb,slip):
    Num_of_states=row*column
    Num_of_actions=5
    model_s={}
    model_sa={}
    model=[model_s,model_sa]
    for s in range(Num_of_states):
        model_s.update({s:['L']})
        if s not in absorb:
            if s >= column:
                model_s[s].append('S')
            if s < (row-1)*column:
                model_s[s].append('N')
            if s % column != 0:
                model_s[s].append('W')
            if (s+1) % column != 0:
                model_s[s].append('E')
        for a in model_s[s]:
            if slip !=0:
                if a == 'N':
                    model_sa.update({(s,a):([slip,1-slip],[s,s+column])})
                if a == 'W':
                    model_sa.update({(s,a):([slip,1-slip],[s,s-1])})
                if a == 'S':
                    model_sa.update({(s,a):([slip,1-slip],[s,s-column])})
                if a == 'E':
                    model_sa.update({(s,a):([slip,1-slip],[s,s+1])})
                if a == 'L':
                    model_sa.update({(s,a):([1],[s])})
            else:
                if a == 'N':
                    model_sa.update({(s,a):([1],[s+column])})
                if a == 'W':
                    model_sa.update({(s,a):([1],[s-1])})
                if a == 'S':
                    model_sa.update({(s,a):([1],[s-column])})
                if a == 'E':
                    model_sa.update({(s,a):([1],[s+1])})
                if a == 'L':
                    model_sa.update({(s,a):([1],[s])})
    return model
