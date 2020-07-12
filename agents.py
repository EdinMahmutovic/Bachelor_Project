import random
import numpy as np
from DeepQNetwork import DeepQNetwork
import torch


class RandomAgent(object):
    def __init__(self):
        pass

    def take_action(self, action_space):
        random_action = np.random.choice(action_space) if action_space.size > 0 else -1
        return random_action


class SOFE(object):
    def __init__(self):
        pass

    def take_action(self, action_space):
        action = action_space[0]
        return action


class FIFO(object):
    def __init__(self):
        pass

    def take_action(self, action_space):
        action = action_space[0]
        return action


class LearningAgent(object):
    def __init__(self, gamma, lr, hidden_size, num_layers, num_features, max_jobs, max_orders, num_processes, n_orders,
                 n_process, batch_size, n_actions, input_size, order_autoencoder, process_autoencoder,
                 max_mem_size=int(10e4), epsilon=1, eps_min=0.05, eps_dec=1e-6):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_features = int(num_features)
        self.input_size = int(input_size)
        self.max_jobs = int(max_jobs)
        self.max_orders = int(max_orders)
        self.num_processes = int(num_processes)
        self.n_orders = int(n_orders)
        self.n_process = int(n_process)
        self.batch_size = int(batch_size)
        self.n_actions = int(n_actions)
        self.max_mem_size = max_mem_size
        self.order_autoencoder = order_autoencoder
        self.process_autoencoder = process_autoencoder
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.mem_counter = 0
        self.target_counter = 0
        self.its_per_target_updates = 1000
        self.action_space = [i for i in range(n_actions)]

        self.Q_eval = DeepQNetwork(input_size=self.input_size * self.max_jobs, lr=self.lr, hidden_size=self.hidden_size,
                                   num_layers=self.num_layers, n_actions=self.n_actions)

        self.Q_target = DeepQNetwork(input_size=self.input_size * self.max_jobs, lr=self.lr, hidden_size=self.hidden_size,
                                     num_layers=self.num_layers, n_actions=self.n_actions)

        self.current_observation_memory = np.zeros((0, self.max_jobs, self.num_features))

        self.state_memory = np.zeros((self.max_jobs, self.max_mem_size, self.max_jobs, self.num_features))
        self.state_seq_length = np.zeros(self.max_mem_size)

        self.new_state_memory = np.zeros((self.max_jobs, self.max_mem_size, self.max_jobs, self.num_features))
        self.new_state_seq_length = np.zeros(self.max_mem_size)

        self.action_memory = np.zeros(self.max_mem_size)
        self.reward_memory = np.zeros(self.max_mem_size)

    def store_transition(self, state, action, new_state, reward, new_observation):

        index = self.mem_counter % self.max_mem_size

        self.current_observation_memory = np.vstack((self.current_observation_memory, state[None]))
        seq_length = self.current_observation_memory.shape[0]
        pad_length = self.max_jobs - seq_length
        padded_state = np.pad(self.current_observation_memory, [(0, pad_length), (0, 0), (0, 0)])
        self.state_memory[:, index, :] = padded_state
        self.state_seq_length[index] = seq_length - 1

        self.action_memory[index] = action

        if new_observation:
            self.current_observation_memory = np.zeros((0, self.max_jobs, self.num_features))

        new_state_memory = np.vstack((self.current_observation_memory, new_state[None]))

        new_seq_length = new_state_memory.shape[0]
        new_pad_length = self.max_jobs - new_seq_length
        padded_new_state = np.pad(new_state_memory, [(0, new_pad_length), (0, 0), (0, 0)])
        self.new_state_memory[:, index, :] = padded_new_state
        self.new_state_seq_length[index] = new_seq_length - 1
        self.reward_memory[index] = reward

        self.mem_counter += 1

    def choose_action(self, observation, legal_action_space):
        if np.random.random() > self.epsilon:
            state = torch.Tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(legal_action_space)

        return action

    def choose_greedy_action(self, observation, legal_action_space):
        state = torch.Tensor(observation).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        actions = torch.argsort(actions, descending=True)
        for action in actions:
            action = action.item()
            if action in legal_action_space:
                break

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return

        if self.target_counter % self.its_per_target_updates == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_counter, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.state_memory[:, batch, :, :]
        order_batch = state_batch[:, :, :, 0]
        process_batch = state_batch[:, :, :, 1]

        order_batch = (np.arange(self.max_orders) == order_batch[..., None]-1).astype(int)
        order_batch = order_batch.reshape(self.max_jobs * self.batch_size * self.max_jobs, self.max_orders)

        process_batch = (np.arange(self.num_processes) == process_batch[..., None] - 1).astype(int)
        process_batch = process_batch.reshape(self.max_jobs * self.batch_size * self.max_jobs, self.num_processes)

        order_batch = (self.order_autoencoder.encoder(torch.Tensor(order_batch).to(self.Q_eval.device))).detach().cpu().clone().numpy()
        process_batch = (self.process_autoencoder.encoder(torch.Tensor(process_batch).to(self.Q_eval.device))).detach().cpu().clone().numpy()

        order_batch = order_batch.reshape(self.max_jobs, self.batch_size, self.max_jobs, self.n_orders)
        process_batch = process_batch.reshape(self.max_jobs, self.batch_size, self.max_jobs, self.n_process)

        state_batch = np.concatenate((order_batch, process_batch, state_batch[:, :, :, 2:]), axis=3)
        state_batch = state_batch.reshape(self.max_jobs, self.batch_size, -1)

        state_batch = torch.Tensor(state_batch).to(self.Q_eval.device)
        state_seq_batch = torch.LongTensor(self.state_seq_length[batch]).to(self.Q_eval.device)

        new_state_batch = self.new_state_memory[:, batch, :, :]
        order_batch = new_state_batch[:, :, :, 0]
        process_batch = new_state_batch[:, :, :, 1]

        order_batch = (np.arange(self.max_orders) == order_batch[..., None]-1).astype(int)
        order_batch = order_batch.reshape(self.max_jobs * self.batch_size * self.max_jobs, self.max_orders)

        process_batch = (np.arange(self.num_processes) == process_batch[..., None] - 1).astype(int)
        process_batch = process_batch.reshape(self.max_jobs * self.batch_size * self.max_jobs, self.num_processes)

        order_batch = (self.order_autoencoder.encoder(torch.Tensor(order_batch).to(self.Q_eval.device))).detach().cpu().clone().numpy()
        process_batch = (self.process_autoencoder.encoder(torch.Tensor(process_batch).to(self.Q_eval.device))).detach().cpu().clone().numpy()

        order_batch = order_batch.reshape(self.max_jobs, self.batch_size, self.max_jobs, self.n_orders)
        process_batch = process_batch.reshape(self.max_jobs, self.batch_size, self.max_jobs, self.n_process)

        new_state_batch = np.concatenate((order_batch, process_batch, new_state_batch[:, :, :, 2:]), axis=3)
        new_state_batch = new_state_batch.reshape(self.max_jobs, self.batch_size, -1)

        new_state_batch = torch.Tensor(new_state_batch).to(self.Q_eval.device)
        new_state_seq_batch = torch.LongTensor(self.new_state_seq_length[batch]).to(self.Q_eval.device)

        reward_batch = torch.Tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        tensor_batch_index = torch.LongTensor(batch_index)

        q_eval = self.Q_eval.forward(state_batch, state_seq_batch, tensor_batch_index)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch, new_state_seq_batch, tensor_batch_index)

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min





