import numpy as np
import gurobipy as gp
from gurobipy import GRB
from graph import Graph


class MDP(object):

    def __init__(self, MDP=None):
        self.MDP = MDP

    def states(self):
    # returns the set of states in the MDP
        states=set(i for i in self.MDP[0])
        return states

    def actions(self):
    # returns the set of actions in the MDP
        actions=set()
        for state in self.MDP[0]:
             actions.update(self.MDP[0][state])
        return actions

    def active_actions(self):
    # returns a dictionary in which the keys are the states and the values are
    # the list of available actions for a given state
        return self.MDP[0]

    def state_action_pairs(self):
    # returns a list of state action pairs in the MDP
        return self.MDP[1].keys()

    def successors(self):
     # returns a dictionary in which the keys are the states and the values are
     # the set of successor states from a given state
        succ={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            succ[pair[0]].update(self.MDP[1][pair][1])
        return succ

    def predecessors(self):
     # returns a dictionary in which the keys are the states and the values are
     # the set of predecessor states from a given state
        pre={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            for succ in self.MDP[1][pair][1]:
                pre[succ].add(pair[0])
        return pre

    def pre_state_action_pair(self):
     # returns a dictionary in which the keys are the states and the values are
     # the set of predeccessor state action pairs from a given state
        pre={i : set() for i in self.states() }
        for pair in self.MDP[1]:
            for succ in self.MDP[1][pair][1]:
                pre[succ].add(pair)
        return pre

    def product_MDP(self, time_horizon, true_goal):

        # For a given MDP (S,A,P) and a time horizon N, the following code constructs
        # the product MDP whose states are S x N. We also update the absorbing and goal states
        num_of_states = len(self.states())
        model_s={}
        model_sa={}
        product_model=[model_s,model_sa]
        for step in range(time_horizon):
            for state in range(num_of_states):
                if step == time_horizon - 1:
                    model_s[state + step * num_of_states] = ['L']
                else:
                    model_s[state + step * num_of_states] = self.MDP[0][state].copy()
                for action in model_s[state + step * num_of_states]:
                    model_sa[( state + step * num_of_states , action)] = (self.MDP[1][(state,action)][0].copy(),self.MDP[1][(state,action)][1].copy())
                    for k,successor in enumerate(self.MDP[1][(state,action)][1]):
                        if step == time_horizon -1:
                            model_sa[( state + step * num_of_states , action)][1][k] = state + step * num_of_states
                        else:
                            model_sa[( state + step * num_of_states , action)][1][k] = successor + (step+1) * num_of_states

        # Absorbing states in the product MDP
        absorb_product = []
        for state in product_model[0].keys():
            if product_model[0][state] == ['L']:
                absorb_product.append(state)

        # Goal states in the product MDP
        true_goals_product = []
        for step in range(time_horizon):
            true_goals_product.append(true_goal + step * num_of_states)

        return product_model, absorb_product, true_goals_product

    def reward_for_reach(self,target):
        # Construct a reward function expressing the reachability to a set of target states
        # Inputs:
        # target: a 'list' of target states
        reach_rew = {}
        for pair in self.MDP[1]:
            if pair[0] not in target and set(self.MDP[1][pair][1]).intersection(set(target)) != set():
                reach_rew[pair] = 0
                for index, succ in enumerate(self.MDP[1][pair][1]):
                    if succ in target:
                        reach_rew[pair] = reach_rew[pair] + self.MDP[1][pair][0][index]
            else:
                reach_rew[pair] = 0
        return reach_rew

    def reward_for_shortest_path(self, goals, one_step_cost):
        # Define the reward function to find the shortest path to each goal
        rewards=[{} for k in range(len(goals))]
        for k in range(len(goals)):
            for state in self.states():
                for act in self.active_actions()[state]:
                    if state != goals[k]:
                        rewards[k][(state,act)] = -1 * one_step_cost
                    else:
                        rewards[k][(state,act)] = 0
        return rewards

    def compute_goal_posteriors(self, prior_goals, init, goals, one_step_cost, discount):

        rewards = self.reward_for_shortest_path(goals, one_step_cost)

        # Compute the probability of targeting a goal from each state
        goal_values = []
        for k in range(len(goals)):
            goal_values.append( self.soft_max_val_iter( rewards[k] , [goals[k]] , discount ) )

        # Compute the posterior probability of hitting a goal from each state
        goal_posteriors=[{} for k in range(len(goals))]
        for state in self.states():
            for act in self.active_actions()[state]:
                for k in range(len(goals)):
                    goal_posteriors[k][(state,act)] = np.exp(goal_values[k][state]-goal_values[k][init])*prior_goals[k]
                denum=np.array([goal_posteriors[i][(state,act)] for i in range(len(goals))])
                for k in range(len(goals)):
                    goal_posteriors[k][(state,act)] = goal_posteriors[k][(state,act)]/ np.sum(denum)
        return goal_posteriors


    def value_evaluate(self,reward ,policy, discount):
        Q_val, V_val_new, V_val, diff = {}, {}, {} , {}
        eps=1e-4
        for pair in self.MDP[1]:
            Q_val[pair] = 0
            V_val[pair[0]], V_val_new[pair[0]] = 0,0
            diff[pair[0]] = 1
        while diff[max(diff, key=diff.get)] > eps:
            for pair in self.MDP[1]:
                succ_sum = 0
                for k,succ in enumerate(self.MDP[1][pair][1]):
                    succ_sum += self.MDP[1][pair][0][k]*V_val[succ]
                Q_val[pair] = reward[pair] + discount * succ_sum
            for state in self.MDP[0]:
                V_val_new[state] = Q_val[(state,policy[state])]
                diff[state] = abs(V_val_new[state] - V_val[state])
                V_val[state] = V_val_new[state]
        #    print(diff[max(diff, key=diff.get)])
        return V_val, Q_val


    def compute_discounted_max_reach(self, init, target, discount):
        Q_val, V_val_new, V_val, diff = {}, {}, {} , {}
        policy = {}
        eps=1e-4
        reward = self.reward_for_reach(target)
        for pair in self.MDP[1]:
            Q_val[pair] = 0
            V_val[pair[0]], V_val_new[pair[0]] = 0,0
            diff[pair[0]] = 1
        while diff[max(diff, key=diff.get)] > eps:
            for pair in self.MDP[1]:
                succ_sum = 0
                for k,succ in enumerate(self.MDP[1][pair][1]):
                    succ_sum += self.MDP[1][pair][0][k]*V_val[succ]
                Q_val[pair] = reward[pair] + discount * succ_sum
            for state in self.MDP[0]:
                Q_array = [Q_val[(state,k)] for k in self.MDP[0][state]]
                V_val_new[state] = max(Q_array)
                opt_action_index = Q_array.index(max(Q_array))
                policy[state] = self.active_actions()[state][opt_action_index]
                diff[state] = abs(V_val_new[state] - V_val[state])
                V_val[state] = V_val_new[state]
        return V_val[init],policy

    def soft_max_val_iter(self, reward, goal, discount):
        V_val_new, V_val, diff = {}, {}, {}
        eps=1e-4
        for state in self.states():
            if state not in goal:
                V_val[state] = float('-inf')
                diff[state] = 1
            else:
                V_val[state] = 0
                diff[state] = 0

        while diff[max(diff, key=diff.get)] > eps:
            for state in self.MDP[0]:
                if state not in goal:
                     V_val_new[state] = float('-inf')
                else:
                    V_val_new[state] = 0

            for state in self.states():
                for act in self.active_actions()[state]:
                    succ_sum = 0
                    for k,succ in enumerate(self.MDP[1][(state,act)][1]):
                        succ_sum += self.MDP[1][(state,act)][0][k]*V_val[succ]
                    V_val_new[state] = self.softmax(V_val_new[state], reward[(state,act)]+discount*succ_sum)

            for state in self.MDP[0]:
                if np.amax([V_val_new[state], V_val[state]]) == float('-inf'):
                    diff[state] = discount * diff[state]
                else:
                    diff[state] = abs(V_val_new[state] - V_val[state])
                V_val[state] = V_val_new[state]
        return V_val

    def softmax(self,x1,x2):
        max_val = np.amax([x1,x2])
        min_val = np.amin([x1,x2])
        if max_val == float('-inf'):
            diff = 0
        else:
            diff = np.exp(min_val - max_val)
        return max_val + np.log(1+diff)

    def compute_max_reach_value_and_policy(self,init, target, absorbing, discount = 1):
     # MDP.compute_max_reach_value_and_policy(init,target) returns
     # (i) the maximum reachability probability to a 'list' of target states from an initial state,
     # (ii) a PROPER policy that maximizes the reachability probability

     # Inputs:
     # init : unique initial state
     # target: the 'list' of ABSORBING target states

        # Define reachability reward
        # Define UNIFORM!!! initial distribution
        alpha = np.ones((len(self.states()),1))
        reach_rew=self.reward_for_reach(target)

        # The following optimization problem is known as the 'dual program'.
        # The variables X(s,a) represent the expected residence time in pair (s,a)
        m = gp.Model("max_reach_value_computation")
        m.setParam( 'OutputFlag', False )
        m2 = gp.Model("max_reach_policy_computation")
        m2.setParam( 'OutputFlag', False )
        X = m.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        X2 = m2.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        pre_lambda_sum, post_lambda_sum, total_reach_prob = {} , {} , gp.LinExpr()
        pre_lambda_sum2, post_lambda_sum2, total_reach_prob2, total_expected  = {}, {} , gp.LinExpr(), gp.LinExpr()
        for state in self.states():
            pre_lambda_sum[state], post_lambda_sum[state] = gp.LinExpr() , gp.LinExpr()
            pre_lambda_sum2[state] , post_lambda_sum2[state] = gp.LinExpr() , gp.LinExpr()

            for act in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(act)
                post_lambda_sum[state].add(X[state,act_ind] , 1)
                total_reach_prob.add(X[state,act_ind] , reach_rew[(state,act)])
                post_lambda_sum2[state].add(X2[state,act_ind] , 1)
                total_reach_prob2.add(X2[state,act_ind] , reach_rew[(state,act)])
                total_expected.add(X2[state,act_ind] , 1)

            for pre in self.pre_state_action_pair()[state]:
                trans_prob_index = self.MDP[1][pre][1].index(state)
                act_index = self.MDP[0][pre[0]].index(pre[1])
                pre_lambda_sum[state].add( X[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])
                pre_lambda_sum2[state].add( X2[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])

            # Flow equation for each state
            if state not in absorbing:
                m.addConstr( post_lambda_sum[state] - discount * pre_lambda_sum[state] == alpha[state])
                m2.addConstr( post_lambda_sum2[state] - discount * pre_lambda_sum2[state] == alpha[state])

        m.setObjective(total_reach_prob, GRB.MAXIMIZE)
        m.optimize()

        policy={}
        for state in range(len(self.states())):
            policy[state]=self.MDP[0][state][0]
            for action in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(action)
                if X[state,act_ind].x >= 1e-4:
                    policy[state] = action
        V_val,_=self.value_evaluate(reach_rew,policy, discount)
        max_reach_val=V_val[init]

        m2.addConstr( total_reach_prob2 >= max_reach_val)
        m2.setObjective( total_expected, GRB.MINIMIZE)
        m2.optimize()
        optimal_policy={}
        for state in range(len(self.states())):
            optimal_policy[state]=self.MDP[0][state][0]
            for action in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(action)
                if X2[state,act_ind].x >= 1e-4:
                    optimal_policy[state] = action

        return max_reach_val, optimal_policy

    def compute_min_cost_subject_to_max_reach(self,init,target,absorbing, cost, discount = 1):
     # MDP.compute_max_reach_value_and_policy(init,target) returns
     # (i) the minimum cost to reach a 'list' of target states with maximum probability,
     # (ii) a policy that minimizes the expected total cost while maximizing the reachability probability

     # Inputs:
     # init: the unique initial state init \in S
     # cost: the 'dictionary' of NONNEGATIVE cost values for each state-action pair

        # Define initial distribution
        alpha = np.zeros((len(self.states()),1))
        alpha[init] = 1

        # Compute maximum reachability probability to the target set
        max_reach_val, _ = self.compute_max_reach_value_and_policy(init,target, absorbing, discount)
        reach_rew=self.reward_for_reach(target)
        print(max_reach_val)
        # The following optimization problem is known as the 'dual program'.
        # The variables X(s,a) represent the expected residence time in pair (s,a)
        m = gp.Model("min_cost_value_computation")
        #m.setParam( 'OutputFlag', False )
        m2 = gp.Model("min_cost_policy_computation")
        m2.setParam( 'OutputFlag', False )
        X = m.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        X2 = m2.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda')
        pre_lambda_sum, post_lambda_sum, total_reach_prob, total_expected_cost= {} , {} , gp.LinExpr(), gp.LinExpr()
        pre_lambda_sum2, post_lambda_sum2, total_reach_prob2 = {}, {} , gp.LinExpr()
        total_expected_cost2, total_expected_time, total_expected_time2 = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()

        for state in self.states():
            pre_lambda_sum[state], post_lambda_sum[state] = gp.LinExpr() , gp.LinExpr()
            pre_lambda_sum2[state] , post_lambda_sum2[state] = gp.LinExpr() , gp.LinExpr()

            for act in self.active_actions()[state]:
                act_ind = self.MDP[0][state].index(act)

                post_lambda_sum[state].add(X[state,act_ind] , 1)
                total_reach_prob.add(X[state,act_ind] , reach_rew[(state,act)])
                total_expected_cost.add(X[state,act_ind] , cost[(state,act)])

                post_lambda_sum2[state].add(X2[state,act_ind] , 1)
                total_reach_prob2.add(X2[state,act_ind] , reach_rew[(state,act)])
                total_expected_cost2.add(X2[state,act_ind] , cost[(state,act)])
                total_expected_time.add(X[state,act_ind] , 1)
                total_expected_time2.add(X2[state,act_ind] , 1)

            for pre in self.pre_state_action_pair()[state]:
                trans_prob_index = self.MDP[1][pre][1].index(state)
                act_index = self.MDP[0][pre[0]].index(pre[1])
                pre_lambda_sum[state].add( X[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])
                pre_lambda_sum2[state].add( X2[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])

            # Flow equation for each state
            if state not in absorbing:
                m.addConstr( post_lambda_sum[state] - discount * pre_lambda_sum[state] == alpha[state])
                m2.addConstr( post_lambda_sum2[state] - discount * pre_lambda_sum2[state] == alpha[state])

        m.addConstr( total_reach_prob >= max_reach_val)
        #m.addConstr( total_expected_time <= 70)
        m.setObjective(total_expected_cost, GRB.MINIMIZE)
        m.optimize()

        min_cost_val=m.objVal

    #    m2.addConstr( total_expected_time2 <= 70)
        m2.addConstr( total_expected_cost2 <= min_cost_val)
        m2.addConstr( total_reach_prob2 >= max_reach_val)
        m2.setObjective( total_expected_time2, GRB.MINIMIZE)
        m2.optimize()
        if discount == 1:
            optimal_policy={}
            for state in range(len(self.states())):
                optimal_policy[state]=self.MDP[0][state][0]
                opt_index = 0
                for action in self.active_actions()[state]:
                    act_ind = self.MDP[0][state].index(action)
                    if X2[state,act_ind].x >= X2[state,opt_index].x:
                        optimal_policy[state]=action
                        opt_index = act_ind
        else:
            optimal_policy={}
            for state in range(len(self.states())):
                optimal_policy[state]=self.MDP[0][state][0]
                opt_index = 0
                for action in self.active_actions()[state]:
                    act_ind = self.MDP[0][state].index(action)
                    if X[state,act_ind].x >= X[state,opt_index].x:
                        optimal_policy[state]=action
                        opt_index = act_ind

        return min_cost_val, optimal_policy

    def MILP_min_discounted_cost_max_reach(self,init,target,absorbing,cost,discount):
        # Define initial distribution
        alpha       = np.zeros((len(self.states()),1))
        alpha[init] = 1

        # Compute maximum reachability probability to the target set
        max_reach_val, _ = self.compute_max_reach_value_and_policy(init,target, absorbing)
        reach_rew        = self.reward_for_reach(target)

        # Define big-M constant
        # !!!! (assumes deterministic transitions for now) !!!!!!
        M = len(self.states())

        # Define the optimization model and the variables
        model    = gp.Model("min_cost_value_computation")
        lambda_1 = model.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda_1')
        lambda_2 = model.addVars(len(self.states()), len(self.actions()), lb=0.0, name='lambda_2')
        Delta    = model.addVars(len(self.states()), len(self.actions()), vtype=GRB.BINARY, name='Delta')


        occupation_1_pre, occupation_1_post      = {} , {}
        occupation_2_pre, occupation_2_post      = {} , {}
        sum_delta                                = {}
        total_reach_prob, total_expected_cost    = gp.LinExpr(), gp.LinExpr()

        for state in self.states():
            occupation_1_pre[state], occupation_1_post[state] = gp.LinExpr() , gp.LinExpr()
            occupation_2_pre[state], occupation_2_post[state] = gp.LinExpr() , gp.LinExpr()
            sum_delta[state]                                  = gp.LinExpr()

            for act_ind, act in enumerate(self.active_actions()[state]):

                occupation_1_post[state].add(lambda_1[state,act_ind] , 1)
                occupation_2_post[state].add(lambda_2[state,act_ind] , 1)
                sum_delta[state].add(Delta[state,act_ind] , 1)

                total_expected_cost.add(lambda_1[state,act_ind] , cost[(state,act)])
                total_reach_prob.add(lambda_2[state,act_ind] , reach_rew[(state,act)])

                model.addConstr(lambda_1[state,act_ind]/M <= Delta[state,act_ind])
                model.addConstr(lambda_2[state,act_ind]/M <= Delta[state,act_ind])

            model.addConstr(sum_delta[state] <= 1)

            for pre in self.pre_state_action_pair()[state]:
                trans_prob_index = self.MDP[1][pre][1].index(state)
                act_index        = self.MDP[0][pre[0]].index(pre[1])

                occupation_1_pre[state].add( lambda_1[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])
                occupation_2_pre[state].add( lambda_2[pre[0],act_index] , self.MDP[1][pre][0][trans_prob_index])

            # Flow equation for each state
            if state not in absorbing:
                model.addConstr( occupation_1_post[state] - discount * occupation_1_pre[state] == alpha[state])
                model.addConstr( occupation_2_post[state] - occupation_2_pre[state] == alpha[state])

        model.addConstr( total_reach_prob >= max_reach_val)
        model.setObjective(total_expected_cost, GRB.MINIMIZE)
        model.optimize()


        optimal_policy={}
        for state in range(len(self.states())):
            optimal_policy[state]=self.MDP[0][state][0]
            opt_index = 0
            for act_ind,action in enumerate(self.active_actions()[state]):
                if Delta[state,act_ind].x >= Delta[state,opt_index].x:
                    optimal_policy[state]=action
                    opt_index = act_ind
        return optimal_policy
