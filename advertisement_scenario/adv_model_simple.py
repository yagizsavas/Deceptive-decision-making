import numpy as np
import time
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

def adv_model(N_states, absorb):
    model_s={}
    model_sa={}
    model=[model_s,model_sa]
    for s in range(N_states):
        model_s.update({s:[0]}) #in all states there's the option not to invest extra
    
        if s not in absorb:
            for j in range(1,4):
                model_s[s].append(j)
            
                
        sigma = 0.05                         
        
        #if i_1 == 0 and j_1 == 20:
            
        scale = 1/N_states*s   #scales {0,...N_states} to [0,1]
        
        for a in model_s[s]:   
            #I guess all the following could be summed up more efficiently
            if a == 0:   
                if s in absorb:
                    model_sa.update({(s,a):([1],[s])}) 
                    continue
                
                # b1 = 0.95 #mu_bounds[a][0]
                # b2 = 0.8 #mu_bounds[a][1]
                # mu = (b2-b1)/(N_states_inv-1)*j_1 + b1   #gaussian mean from range [0,20] to range [0.95,0.8]
                
                # cur_p = 0.85
                                           
                if s == 0:                    
                    model_sa.update({(s,a):([1],[s])}) 
                else:
                    model_sa.update({(s,a):([ (.5 - scale*0.1), (.5 - scale*0.15), scale*0.25 ],[s, s-1, s+1])})
                    
             
            elif a == 1:   #similar, just give more probability to user increase + extra level            
            
                if s == 0:                    
                    model_sa.update({(s,a):([ .75, .25],[s, s+1,])}) 
                else:
                    model_sa.update({(s,a):([ (.35 - scale*0.3), (.25 - scale*0.2), 0.4 + scale*0.35,  scale*0.15 ],[s, s-1, s+1, s+2])})
                                         
                
            elif a == 2:
            
                if s == 0:                    
                    model_sa.update({(s,a):([0.5, 0.5],[s, s+1])}) 
                else:
                    #model_sa.update({(s,a):([ (.25 - scale*0.25), (.15 - scale*0.15),  0.5 + scale*0.3, 0.1 + scale*0.1],[s, s-1, s+1, s+2])})
                    model_sa.update({(s,a):([ (.2 - scale*0.2), (.1 - scale*0.1),  0.5 + scale*0.2, 0.2 + scale*0.1],[s, s-1, s+1, s+2])})

            
            else:
                
                if s == 0:                    
                    model_sa.update({(s,a):([0.1, 0.6, 0.3],[s, s+1,s+2])}) 
                else:
                    model_sa.update({(s,a):([ (.1 - scale*0.1), (.05 - scale*0.05),  0.55 + scale*0.1, 0.25 + scale*0.03,  0.05 + scale*0.02],[s, s-1, s+1, s+2, s+3])})                
             
          
                    
    # for s in range(N_states):
    #     if s not in absorb:
    #         #print('\n')
    #         #print('state:', s)
    #         flag = 0
    #         for a in model[0][s]:
    #             Post = model[1][(s,a)][1]
    #             if len(Post) != 1:
    #                 flag = 1
    #         if flag==0:
    #             print(s)
                #for s_post in Post:
                #    if s_post ==1778:#in absorb:
                #        i_1 = np.floor_divide(s,N_states_inv)
                #        j_1 = s - N_states_inv*i_1 
                #        print('state:', s, '=', i_1,j_1)
                #        print('action:', a)
                #        print( model[1][(s,a)] )
                  #      print('\n')
                #print('action:', a)
                #print( model[1][(s,a)]  )
                
# for s in range(N_states):    
#     for a in range(0, len(model[0][s])):
#         #print('action = ', a)
#         sum = 0
#         for q in range(0, len(model[1][(s,a)][0])):
#             if model[1][(s,a)][0][q] < 0:
#                 print('q = ', q)
#                 print( model[1][(s,a)][0][q] ) 
#                 time.sleep(1)
                
#             sum = sum + model[1][(s,a)][0][q]
#         print(sum)


    return model
