import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))

class circulant_dynamics():
  def __init__(self):
    self.s = np.array([1,2,3,4,5])
    self.P0 = np.array([[1/10, 9/10, 0, 0, 0],
                        [1/10, 0, 9/10, 0, 0],
                        [1/10, 0, 0, 9/10, 0],
                        [1/10, 0, 0, 0, 9/10],
                        [1/10, 0, 0, 0, 9/10]])
    self.P1 = np.array([[1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0]])
    self.R0 = np.array([pow(0.9,1),pow(0.9,2),pow(0.9,3),pow(0.9,4),pow(0.9,5)])
    self.R1 = np.array([0,0,0,0,0])

  def new_state(self,state,action):
    if int(action) == 0:
      x = np.random.choice(self.s,p=self.P0[int(state)-1])
    elif int(action) == 1:
      x = np.random.choice(self.s,p=self.P1[int(state)-1])
    return x

  def reward(self,state,action):
    if int(action) == 0:
      reward = self.R0[int(state)-1]
    elif int(action) == 1:
      reward = self.R1[int(state)-1]
    return reward

  def get_transition(self,state,action):
    reward = self.reward(state,action)
    nstate = self.new_state(state,action)
    return Transition(state,action,reward,nstate)

  def get_transitions(self,states,actions):
    # returns tuples of transition for given states and actions
    # transition -> (s_t,a_t,r_t,s_(t+1))
    transitions = (self.get_transition(states[0],actions[0]),)

    for (s,a) in list(zip(np.delete(states,0),np.delete(actions,0))):
       tran = self.get_transition(s,a)
       transitions = transitions + (tran, )

    return tuple(set(transitions))