import time
from helping_functions import *
import pickle

'''
This script is the main script that initializes the production environment with the learning agent and trains it. 
'''


#  Giving the environment that the agent has to train on.
env_production = "18p4m/"
env_id = "18p4m"
products = env_production + "products.csv"
machines = env_production + "machines.csv"
processes = env_production + "processes.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  initialize the maximum values of orders and jobs as well as the minimum starting job list size.
max_orders = 30
max_jobs = 50
min_size = 20

#  Initializing the environment.
env = Environment(environment_machines=machines, environment_products=products, environment_processes=processes,
                  min_size=min_size, max_jobs=max_jobs, max_orders=max_orders)

#  Computing and defining the parameters and hyper-parameters.
num_features = env.num_features
num_processes = env.num_processes
hidden_size = num_features

training_time = 0.8  # training time in hours
end_time = time.time() + 30  # training_time * 60 * 60
gamma = 0.95
alpha = 0.001
lr_auto_encoder = 0.001
num_layers = 2
batch_size = 32
n_actions = env.num_machines * max_jobs
max_mem_size = 10 ** 6
epsilon = 1
eps_min = 0.05
eps_dec = 1e-5
test_period = 5

#  Saving a tuple of information of the environment which will be passed into some helping functions.
env_info = (machines, products, processes, min_size, max_jobs, max_orders, num_features, num_processes)

#  Decide if the encoders needs to be trained and whether the agent has been trained before on the given environment.
train_order_encoder = True
train_process_encoder = True
pretrained_agent = False

#  Initializing the auto encoders.
order_autoencoder, n_orders = get_order_autoencoder(max_orders=max_orders, train_order_encoder=train_order_encoder)
process_autoencoder, n_process = get_process_autoencoder(num_processes=num_processes,
                                                         train_process_encoder=train_process_encoder)

#  Save the auto encoders in a tuple for use in helping functions.
encoders = (order_autoencoder, process_autoencoder)
input_dim = n_orders + n_process + 3  # The dimensions of a job vector after dimension reduction with use of AE.

#  Initialize the learning agent.
agent = LearningAgent(gamma=gamma, lr=alpha, hidden_size=hidden_size, num_layers=num_layers, num_features=num_features,
                      max_jobs=max_jobs, max_orders=max_orders, input_size=input_dim, batch_size=batch_size,
                      n_actions=n_actions, order_autoencoder=order_autoencoder, process_autoencoder=process_autoencoder,
                      max_mem_size=max_mem_size, num_processes=num_processes, n_orders=n_orders, n_process=n_process,
                      epsilon=epsilon, eps_min=eps_min, eps_dec=eps_dec)

#  If a pre-trained version of the agent exists, the weights and biases of the DQN are read and initialized.
if pretrained_agent:
    # state_dict = torch.load("q_model" + env_id + ".pt")
    # agent.Q_eval.load_state_dict(state_dict)
    agent = pickle.load(open("agent.pickle", "rb", -1))
    agent.reset()

#  Initialize the environment state.
state = env.state
new_observation = True
num_jobs = []

its = 0

performance = np.zeros((0, 4, test_period * 365))
#  Training the agent for a period of time.
while time.time() < end_time:
    if new_observation:
        observation_memory = np.zeros((0, max_jobs, num_features))
        num_jobs.append(len(env.jobs.list))
        # print("iteration::", env.t)

    input_mat, observation_memory = memory_processing(observation_memory, state, env_info, encoders)
    action_space = env.action_space
    action = agent.choose_action(input_mat, action_space)
    new_state, reward, new_observation = env.do_action(action)
    agent.store_transition(state=state, action=action, new_state=new_state,
                           reward=reward, new_observation=new_observation)
    agent.learn()
    state = new_state

    if its % 1000 == 0:
        test_performance = test_agent(learning_agent=agent, env_info=env_info, encoders=encoders,
                                      test_period=test_period)
        performance = np.vstack((performance, test_performance[None]))
    its += 1

with open("agent.pickle", "wb") as file_:
    pickle.dump(agent, file_, -1)

np.save("test_performance.npy", performance)
print("Number of iterations on", training_time, "hours::", env.t)
#torch.save(agent.Q_eval.state_dict(), "q_model" + env_id + ".pt")
