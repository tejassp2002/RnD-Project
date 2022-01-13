from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
import torch
import numpy as np
import pyspiel

import collections
import math
import random
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_spiel.python import rl_agent
import os

from torch.utils.tensorboard import SummaryWriter



# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent implemented in PyTorch."""



Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class ReplayBuffer(object):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def __init__(self, replay_buffer_capacity):
    self._replay_buffer_capacity = replay_buffer_capacity
    self._data = []
    self._next_entry_index = 0

  def add(self, element):
    """Adds `element` to the buffer.

    If the buffer is full, the oldest element will be replaced.

    Args:
      element: data to be added to the buffer.
    """
    if len(self._data) < self._replay_buffer_capacity:
      self._data.append(element)
    else:
      self._data[self._next_entry_index] = element
      self._next_entry_index += 1
      self._next_entry_index %= self._replay_buffer_capacity

  def sample_FGDQN(self,state,action):
    """
    Returns samples with same state and action uniformly sampled
    from the buffer.
    Args:
    """
    data = [self._data[i] for i in range(len(self._data)) if \
      (self._data[i].info_state,self._data[i].action)==(state,action)]


    # print("Number of FGDQN Samples: ",len(data))
    if len(data)>32:
      data = random.sample(data,32)
    return data


  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()
    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer
    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x


class FGDQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in PyTorch.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(self,
               player_id,
               state_representation_size,
               num_actions,
               fixed_state,
               hidden_layers_sizes=128,
               replay_buffer_capacity=10000,
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=1e-3,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e6),
               optimizer_str="rmsprop",
               loss_str="mse"):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()

    self.player_id = player_id
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration

    # TODO(author6) Allow for optional replay buffer config.
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0
    self._loss_counter = 0
    self.running_loss = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Create the Q-network instances
    self._q_network = MLP(state_representation_size, self._layer_sizes,
                          num_actions).to(device)

    self.fixed_state = torch.Tensor(fixed_state).to(device)

    if loss_str == "mse":
      self.loss_class = F.mse_loss
    elif loss_str == "huber":
      self.loss_class = F.smooth_l1_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    if optimizer_str == "adam":
      self._optimizer = torch.optim.Adam(
          self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = torch.optim.SGD(
          self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == "rmsprop":
      self._optimizer = torch.optim.RMSprop(
          self._q_network.parameters(), lr=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

  def step(self, time_step, is_evaluation=False, add_transition_record=True):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      epsilon = self._get_epsilon(is_evaluation)
      action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._step_counter += 1
      if self._step_counter % self._learn_every == 0:
         loss_per_iter = self.learn()
         if loss_per_iter is not None:
           self._loss_counter += 1
           self.running_loss += loss_per_iter.item()
           if self._loss_counter % running_time == 0:
             print(f"Training Loss {self._loss_counter}: {self.running_loss/running_time}")
             writer.add_scalar("Training Loss", self.running_loss/running_time, self._loss_counter)
             self.running_loss = 0.0

      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        self.add_transition(self._prev_timestep, self._prev_action, time_step)

      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    legal_actions = (time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            prev_time_step.observations["info_state"][self.player_id][:]),
        action=prev_action,
        reward=time_step.rewards[self.player_id],
        next_info_state=time_step.observations["info_state"][self.player_id][:],
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask)
    self._replay_buffer.add(transition)

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:
      info_state = torch.Tensor(np.reshape(info_state, [1, -1]))
      q_values = self._q_network(info_state.to(device)).detach()[0]
      legal_q_values = q_values[legal_actions]
      action = legal_actions[torch.argmax(legal_q_values)]
      probs[action] = 1.0
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer.sample(self._batch_size)

    loss_per_iter = 0

    for i in range(len(transitions)):
        # from the given batch B, get a transition and sample only transitions from the replay buffer
        # with the same state and action pair as of this transition.
        new_transitions = self._replay_buffer.sample_FGDQN(transitions[i].info_state,transitions[i].action)
        # new_transitions are the transitions with the same state and action pair.
        info_states = torch.Tensor(np.array([t.info_state for t in new_transitions])).to(device) #[B, 1741]
        actions = torch.LongTensor(np.array([t.action for t in new_transitions]))  #[B]
        rewards = torch.Tensor(np.array([(a+t.reward)/(b-a) for t in new_transitions])).to(device)
        next_info_states = torch.Tensor(np.array([t.next_info_state for t in new_transitions])).to(device)
        are_final_steps = torch.Tensor(np.array([t.is_final_step for t in new_transitions])).to(device)
        legal_actions_mask = torch.Tensor(np.array([t.legal_actions_mask for t in new_transitions]))

        self._q_values = self._q_network(info_states)
        self._next_q_values = self._q_network(next_info_states)  #[B,205]

        illegal_actions = 1 - legal_actions_mask
        illegal_logits = (illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY).to(device)
        max_next_q = torch.max(self._next_q_values + illegal_logits, dim=1)[0]

        self.fixed_q_value = self._q_network(self.fixed_state)
        #1 torch.Size([1, 1, 205])

        self.fixed_q_value = torch.max(self.fixed_q_value,dim=1)[0]  #maxu Q(i0, u)

        target = (
            rewards + (1 - are_final_steps) * (max_next_q-self.fixed_q_value))

        action_indices = torch.stack([torch.arange(self._q_values.shape[0], dtype=torch.long), actions],dim=0)  #[2,B]
        predictions = self._q_values[list(action_indices)] #[B]

        
        diff = torch.mean(target-predictions)
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
        avg_part = diff.detach()
        loss =  torch.mul(avg_part,diff)
        
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._q_network.parameters():
          param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        actual_loss = self.loss_class(predictions, target) # mse loss
        loss_per_iter += actual_loss

    # if self._step_counter% 50 == 0:
    #     print(f"Q Values {torch.max(self._q_network(torch.Tensor(self.fixed_state).to(device)),dim=1)[0]}")

    return loss_per_iter/len(transitions)

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter



def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def pt_main(game,
            checkpoint_dir,
            num_train_episodes,
            eval_every,
            save_every,
            hidden_layers_sizes,
            replay_buffer_capacity,
            batch_size,
            learn_every):
  env = rl_environment.Environment(game)
  num_players = env.num_players
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]
  hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]
  # pylint: disable=g-complex-comprehension
  # print(learn_every)
  agents = [
      FGDQN(
          player_id=idx,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          fixed_state=np.array([env.reset()[0]["info_state"][idx]]),
          hidden_layers_sizes=hidden_layers_sizes,
          replay_buffer_capacity=replay_buffer_capacity,
          batch_size=batch_size,
          learn_every=learn_every) for idx in range(num_players)
  ]
  reward_counter = 0
  for ep in range(num_train_episodes):

    if (ep + 1) % eval_every == 0:
      reward_counter += 1
      r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
      if isinstance(r_mean, np.ndarray):
        for i in range(len(r_mean)):
          writer.add_scalar(f"Reward{i}", r_mean[i], reward_counter)
      else:
        writer.add_scalar("Reward", r_mean, reward_counter)
      print(f"Reward {reward_counter}: {r_mean}")

    if (ep + 1) % save_every == 0:
      print(f"saving the model at {ep+1} iteration")
      for i in range(num_players):
        torch.save(agents[i]._q_network.state_dict(), f"{checkpoint_dir}/Player{i}_Epoch{ep+1}.pth")

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if env.is_turn_based:
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
      else:
        agents_output = [agent.step(time_step) for agent in agents]
        action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)

  print("saving final model")
  for i in range(num_players):
    torch.save(agents[i]._q_network.state_dict(), f"{checkpoint_dir}/Player{i}_Final.pth")



if __name__ == '__main__':
    game = "catch"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device}!")
    logging_dir = f"./openspiel/{game}/logs"
    writer = SummaryWriter(log_dir=logging_dir)
    checkpoint_dir = f"./openspiel/{game}/checkpoints"
    num_train_episodes = 5000
    eval_every = 20
    save_every = 400
    learn_every = 5
    hidden_layers_sizes = [256, 64]
    replay_buffer_capacity = int(1e5)
    batch_size = 32
    running_time = 200

    #parameters for reward scaling
    tp = pyspiel.load_game(game)
    a = tp.min_utility()
    b = tp.max_utility()

    if not os.path.isdir(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    pt_main(game,
            checkpoint_dir,
            num_train_episodes,
            eval_every,
            save_every,
            hidden_layers_sizes,
            replay_buffer_capacity,
            batch_size,
            learn_every)