from Autoencoder import *
from copy import deepcopy
from model import *
from agents import *


def test_agent(learning_agent, env_info, encoders, test_period=5):
    machines, products, processes, min_size, max_jobs, max_orders, num_features, _ = env_info

    test_env = Environment(environment_machines=machines, environment_products=products,
                           environment_processes=processes,
                           min_size=min_size, max_jobs=max_jobs, max_orders=max_orders)

    random_agent = RandomAgent()
    fifo_agent = FIFO()
    sofe_agent = SOFE()

    learning_agent_env = deepcopy(test_env)
    fifo_env = deepcopy(test_env)
    sofe_env = deepcopy(test_env)
    random_env = deepcopy(test_env)

    envs = [learning_agent_env, fifo_env, sofe_env, random_env]
    agents = [learning_agent, fifo_agent, sofe_agent, random_agent]

    its = 0
    test_performance = np.zeros((len(agents), test_period * 365))
    while its < test_period * 365:

        #print([len(env.orders.list) for env in envs])
        #print([len(env.jobs.list) for env in envs])
        print("Test period::", its)
        for i, (agent, env) in enumerate(zip(agents, envs)):
            observation_memory = np.zeros((0, max_jobs, num_features))
            new_observation = False

            while not new_observation:
                if i == 0:
                    state = env.state
                    input_mat, observation_memory = memory_processing(observation_memory, state, env_info, encoders)
                    action_space = env.action_space
                    action = agent.choose_greedy_action(input_mat, action_space) if action_space.size > 0 else -1
                elif i == 1:
                    action_space = env.orders_by_creation_actions()
                    action = agent.take_action(action_space) if action_space.size > 0 else -1
                elif i == 2:
                    action_space = env.orders_by_deadline_actions()
                    action = agent.take_action(action_space) if action_space.size > 0 else -1
                else:
                    action_space = env.action_space
                    action = agent.take_action(action_space) if action_space.size > 0 else -1

                _, reward, new_observation = env.do_action(action, test=1)
            test_performance[i, its] = reward

        new_orders = test_env.generate_new_orders(universal=True)
        [env.update_order_list(deepcopy(new_orders), test=True) for env in envs]

        its += 1

    return test_performance


def memory_processing(memory, current_state, env_info, encoders):  # Put into a helping functions script

    order_autoencoder, process_autoencoder = encoders
    device = order_autoencoder.device
    machines, products, processes, min_size, max_jobs, max_orders, num_features, num_processes = env_info

    memory = np.vstack((memory, current_state[None]))
    memory_orders = memory[:, :, 0]
    memory_processes = memory[:, :, 1]

    memory_orders = (np.arange(max_orders) == memory_orders[..., None] - 1).astype(int)
    memory_processes = (np.arange(num_processes) == memory_processes[..., None] - 1).astype(int)

    memory_orders = memory_orders.reshape(memory.shape[0] * max_jobs, max_orders)
    memory_processes = memory_processes.reshape(memory.shape[0] * max_jobs, num_processes)

    memory_order_input = (
        order_autoencoder.encoder(torch.Tensor(memory_orders).to(device))).detach().cpu().clone().numpy()
    memory_process_input = (
        process_autoencoder.encoder(torch.Tensor(memory_processes).to(device))).detach().cpu().clone().numpy()

    n_orders = memory_order_input.shape[1]
    n_process = memory_process_input.shape[1]
    input_dim = n_orders + n_process + 3

    memory_order_input = memory_order_input.reshape(memory.shape[0], max_jobs, n_orders)
    memory_process_input = memory_process_input.reshape(memory.shape[0], max_jobs, n_process)

    input_matrix = np.concatenate((memory_order_input, memory_process_input, memory[:, :, 2:]), axis=2)
    input_matrix = input_matrix.reshape(memory.shape[0], 1, max_jobs * input_dim)

    return input_matrix, memory


def get_order_autoencoder(max_orders, train_order_encoder, batch_size=32, lr_auto_encoder=0.001):
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
        order_autoencoder = train_autoencoder(max_size=max_orders, autoencoder=order_autoencoder,
                                              batch_size=batch_size, encoder_type="orders")
    else:
        trained_state_dict = torch.load("auto_encoder_orders" + str(max_orders) + ".pt",
                                        map_location=torch.device('cpu'))
        order_autoencoder.load_state_dict(trained_state_dict)

    return order_autoencoder, n_orders


def get_process_autoencoder(num_processes, train_process_encoder, batch_size=32, lr_auto_encoder=0.01):
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
        process_autoencoder = train_autoencoder(max_size=num_processes, autoencoder=process_autoencoder,
                                                batch_size=batch_size, encoder_type="process")
    else:
        trained_state_dict = torch.load("auto_encoder_process" + str(num_processes) + ".pt",
                                        map_location=torch.device('cpu'))
        process_autoencoder.load_state_dict(trained_state_dict)

    return process_autoencoder, n_process
