from agents import *
from model import *
import time
from Autoencoder import AutoEncoder
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

env_production = "18p4m/"
env_id = "18p4m"
products = env_production + "products.csv"
machines = env_production + "machines.csv"
processes = env_production + "processes.csv"

max_orders = 30
max_jobs = 50
min_size = 20

env = Environment(environment_machines=machines, environment_products=products, environment_processes=processes,
                  min_size=min_size, max_jobs=max_jobs, max_orders=max_orders)

num_features = env.num_features
num_processes = env.num_processes
hidden_size = num_features

training_time = 20  # training time in hours
end_time = time.time() + 60  # training_time * 60 * 60
gamma = 0.95
alpha = 0.0001
num_layers = 2
batch_size = 32
n_actions = env.num_machines * max_jobs
max_mem_size = 10 ** 6
epsilon = 0.9
eps_min = 0.05
eps_dec = 1e-5

train_order_encoder = False
train_process_encoder = False
pretrained_agent = False

lr_auto_encoder = 0.0001
n1_orders = np.ceil(0.8 * max_orders).astype(int)
n2_orders = np.ceil(0.6 * max_orders).astype(int)
n3_orders = np.ceil(0.4 * max_orders).astype(int)
n4_orders = np.ceil(0.2 * max_orders).astype(int)
n5_orders = np.ceil(0.175 * max_orders).astype(int)
n6_orders = np.ceil(0.15 * max_orders).astype(int)
n7_orders = np.ceil(0.1 * max_orders).astype(int)
n_orders = np.ceil(0.08 * max_orders).astype(int)

order_autoencoder = AutoEncoder(n_input=max_orders, n1=n1_orders, n2=n2_orders, n3=n3_orders,
                                n4=n4_orders, n5=n5_orders, n6=n6_orders, n7=n7_orders,
                                n_dim=n_orders, lr=lr_auto_encoder)

if train_order_encoder:
    accuracy_threshold = 0.9995
    test_accuracy_orders = []
    while True:
        n_samples_orders = 10 ** 5
        X_orders = np.eye(max_orders)[np.random.choice(max_orders, n_samples_orders)]
        X_train, X_val = train_test_split(X_orders, test_size=0.2)
        X_train, X_test = train_test_split(X_train, test_size=0.2)

        X_test = torch.Tensor(X_test).to(order_autoencoder.device)
        X_val = torch.Tensor(X_val).to(order_autoencoder.device)
        data_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True,
                                 pin_memory=True)

        for i, x_train in enumerate(iter(data_loader)):
            order_autoencoder.train()
            x_train = x_train.to(order_autoencoder.device)
            order_autoencoder.optimizer.zero_grad()
            decoded_x = order_autoencoder.encoder(x=x_train.float())
            encoded_x = order_autoencoder.decoder(x=decoded_x)

            loss = order_autoencoder.loss(encoded_x, torch.argmax(x_train, dim=1))
            loss.backward()
            order_autoencoder.optimizer.step()

            if i % 5000 == 0:
                order_autoencoder.eval()
                with torch.no_grad():
                    X_test = X_test.to(order_autoencoder.device)
                    x_encoded = order_autoencoder.encoder(x=X_test)
                    x_decoded = order_autoencoder.decoder(x=x_encoded)

                    max_index = x_decoded.max(dim=1)[1]
                    test_accuracy = (max_index == torch.argmax(X_test, dim=1)).sum()
                    test_accuracy = test_accuracy.item() / X_test.shape[0]
                    test_accuracy_orders.append(test_accuracy)

        order_autoencoder.eval()
        with torch.no_grad():
            x_encoded = order_autoencoder.encoder(x=X_val)
            x_decoded = order_autoencoder.decoder(x=x_encoded)

            max_index = x_decoded.max(dim=1)[1]
            val_accuracy = (max_index == torch.argmax(X_val, dim=1)).sum()
            val_accuracy = val_accuracy.item() / X_val.shape[0]

            if val_accuracy > accuracy_threshold:
                torch.save(order_autoencoder.state_dict(), "auto_encoder_orders" + str(max_orders) + ".pt")
                with open('order_accuracy' + str(max_orders) + '.pickle', 'wb') as b:
                    pickle.dump(test_accuracy_orders, b)

                break

else:
    trained_state_dict = torch.load("auto_encoder_orders" + str(max_orders) + ".pt")
    order_autoencoder.load_state_dict(trained_state_dict)

n1_process = np.ceil(0.80 * num_processes).astype(int)
n2_process = np.ceil(0.60 * num_processes).astype(int)
n3_process = np.ceil(0.40 * num_processes).astype(int)
n4_process = np.ceil(0.35 * num_processes).astype(int)
n5_process = np.ceil(0.30 * num_processes).astype(int)
n6_process = np.ceil(0.25 * num_processes).astype(int)
n7_process = np.ceil(0.20 * num_processes).astype(int)
n_process = np.ceil(0.15 * num_processes).astype(int)

process_autoencoder = AutoEncoder(n_input=num_processes, n1=n1_process, n2=n2_process, n3=n3_process,
                                  n4=n4_process, n5=n5_process, n6=n6_process, n7=n7_process,
                                  n_dim=n_process, lr=lr_auto_encoder)

if train_process_encoder:
    accuracy_threshold = 0.9995
    test_accuracy_process = []
    while True:
        n_samples_process = 10 ** 5
        X_process = np.eye(num_processes)[np.random.choice(num_processes, n_samples_process)]
        X_train, X_val = train_test_split(X_process, test_size=0.2)
        X_train, X_test = train_test_split(X_train, test_size=0.2)

        X_test = torch.Tensor(X_test).to(process_autoencoder.device)
        X_val = torch.Tensor(X_val).to(process_autoencoder.device)
        data_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True,
                                 pin_memory=True)

        for i, x_train in enumerate(iter(data_loader)):
            process_autoencoder.train()
            x_train = x_train.to(process_autoencoder.device)
            process_autoencoder.optimizer.zero_grad()
            decoded_x = process_autoencoder.encoder(x=x_train.float())
            encoded_x = process_autoencoder.decoder(x=decoded_x)

            loss = process_autoencoder.loss(encoded_x, torch.argmax(x_train, dim=1))
            loss.backward()
            process_autoencoder.optimizer.step()

            if i % 5000 == 0:
                process_autoencoder.eval()
                with torch.no_grad():
                    X_test = X_test.to(process_autoencoder.device)
                    x_encoded = process_autoencoder.encoder(x=X_test)
                    x_decoded = process_autoencoder.decoder(x=x_encoded)

                    max_index = x_decoded.max(dim=1)[1]
                    test_accuracy = (max_index == torch.argmax(X_test, dim=1)).sum()
                    test_accuracy = test_accuracy.item() / X_test.shape[0]
                    test_accuracy_process.append(test_accuracy)

        process_autoencoder.eval()
        with torch.no_grad():
            x_encoded = process_autoencoder.encoder(x=X_val)
            x_decoded = process_autoencoder.decoder(x=x_encoded)

            max_index = x_decoded.max(dim=1)[1]
            val_accuracy = (max_index == torch.argmax(X_val, dim=1)).sum()
            val_accuracy = val_accuracy.item() / X_val.shape[0]

            if val_accuracy > accuracy_threshold:
                torch.save(process_autoencoder.state_dict(), "auto_encoder_process" + str(num_processes) + ".pt")
                with open('process_accuracy' + str(num_processes) + '.pickle', 'wb') as b:
                    pickle.dump(test_accuracy_process, b)

                break

else:
    trained_state_dict = torch.load("auto_encoder_process" + str(num_processes) + ".pt")
    process_autoencoder.load_state_dict(trained_state_dict)


input_dim = n_orders + n_process + 3


agent = LearningAgent(gamma=gamma, lr=alpha, hidden_size=hidden_size, num_layers=num_layers, num_features=num_features,
                      max_jobs=max_jobs, max_orders=max_orders, input_size=input_dim, batch_size=batch_size,
                      n_actions=n_actions, order_autoencoder=order_autoencoder, process_autoencoder=process_autoencoder,
                      max_mem_size=max_mem_size, num_processes=num_processes, n_orders=n_orders, n_process=n_process,
                      epsilon=epsilon, eps_min=eps_min, eps_dec=eps_dec)

if pretrained_agent:
    state_dict = torch.load("q_model" + env_id + ".pt")
    agent.Q_eval.load_state_dict(state_dict)

avg_delays7 = np.array([])
avg_delays30 = np.array([])
avg_delays60 = np.array([])
avg_delays90 = np.array([])
avg_delays180 = np.array([])
delays = np.zeros(180)
delay_length = 180

its = 0
rewards = []

state = env.state
observation_memory = np.zeros((0, state.shape[0], state.shape[1]))
input_mat = np.zeros((0, max_jobs * input_dim))
new_observation = True

while time.time() < end_time:
    if new_observation:
        observation_memory = np.zeros((0, max_jobs, num_features))
        input_mat = np.zeros((0, max_jobs * input_dim))

    observation_memory = np.vstack((observation_memory, state[None]))
    orders = observation_memory[:, :, 0]
    processes = observation_memory[:, :, 1]

    orders = (np.arange(max_orders) == orders[..., None] - 1).astype(int)
    processes = (np.arange(num_processes) == processes[..., None] - 1).astype(int)

    orders = orders.reshape(observation_memory.shape[0] * max_jobs, max_orders)
    processes = processes.reshape(observation_memory.shape[0] * max_jobs, num_processes)

    order_input = (order_autoencoder.encoder(torch.Tensor(orders))).detach().numpy()
    process_input = (process_autoencoder.encoder(torch.Tensor(processes))).detach().numpy()

    order_input = order_input.reshape(observation_memory.shape[0], max_jobs, n_orders)
    process_input = process_input.reshape(observation_memory.shape[0], max_jobs, n_process)

    input_mat = np.concatenate((order_input, process_input, observation_memory[:, :, 2:]), axis=2)
    input_mat = input_mat.reshape(observation_memory.shape[0], 1, max_jobs * input_dim)
    action_space = env.action_space
    action = agent.choose_action(input_mat, action_space)
    new_state, reward, new_observation = env.do_action(action)

    if new_observation:
        rewards.append(reward)

        delays = np.append(delays, reward)
        delays = delays[-delay_length:]
        avg_delays7 = np.append(avg_delays7, np.mean(delays[-7:]))
        avg_delays30 = np.append(avg_delays7, np.mean(delays[-30:]))
        avg_delays60 = np.append(avg_delays7, np.mean(delays[-60:]))
        avg_delays90 = np.append(avg_delays7, np.mean(delays[-90:]))
        avg_delays180 = np.append(avg_delays7, np.mean(delays[-180:]))

    agent.store_transition(state=state, action=action, new_state=new_state,
                           reward=reward, new_observation=new_observation)
    agent.learn()
    state = new_state
    its += 1

with open('rewards.pkl', 'wb') as f:
    pickle.dump(rewards, f)

with open('avg_delays7.pkl', 'wb') as f:
    pickle.dump(avg_delays7, f)

with open('avg_delays30.pkl', 'wb') as f:
    pickle.dump(avg_delays7, f)

with open('avg_delays60.pkl', 'wb') as f:
    pickle.dump(avg_delays7, f)

with open('avg_delays90.pkl', 'wb') as f:
    pickle.dump(avg_delays7, f)

with open('avg_delays180.pkl', 'wb') as f:
    pickle.dump(avg_delays7, f)

print("Number of iterations on", training_time, "hours::", its)
torch.save(agent.Q_eval.state_dict(), "q_model.pt")
