import os
from model import *
from agents import *
import matplotlib.pyplot as plt
from DeepQNetwork import DeepQNetwork
import torch.nn as nn
import time


class Nothing(object):
    def __init__(self):
        state_batch = self.state_memory[:, batch, :]
        state_input_batch = torch.zeros(self.max_jobs, 0, self.input_size)

        for b in range(self.batch_size):
            order_id = self.state_memory[:, b, :, 0]
            order_vector = torch.zeros(self.seq_length, self.max_jobs, self.max_orders)
            order_vector[seq_index, order_id] = 1

            process_id = self.state_memory[:, b, 1]
            process_vector = torch.zeros(self.seq_length, self.max_jobs, self.num_processes)
            process_vector[seq_index, process_id] = 1

            order_input = self.order_autoencoder(self.state_memory[:, b, 0])
            process_input = self.process_autoencoder(self.state_memory[:, b, 1])
            feature_input = self.state_memory[:, b, 2:]

            state_input = torch.cat((order_input, process_input, feature_input), dim=1)
            state_input = state_input.unsqueeze(dim=1)
            state_input_batch = torch.cat((state_input_batch, state_input), dim=1)