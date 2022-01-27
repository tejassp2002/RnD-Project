import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))

class circulant_dynamics1():
  def __init__(self):
    self.s = np.array([1,2,3,4])
    self.P0 = np.array([[1/2, 0, 0, 1/2],
                        [1/2, 1/2, 0, 0],
                        [0, 1/2, 1/2, 0],
                        [0, 0, 1/2, 1/2]])
    self.P1 = np.transpose(self.P0)
    self.R = np.array([-1,0,0,1])

  def new_state(self,state,action):
    if int(action) == 0:
      x = np.random.choice(self.s,p=self.P0[int(state)-1])
    elif int(action) == 1:
      x = np.random.choice(self.s,p=self.P1[int(state)-1])
    return x

  def reward(self,state,action):
    return self.R[int(state)-1]

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
