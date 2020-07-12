import numpy as np
import pandas as pd
import operator

'''
This script contains all components in a production; machines, products, processes, orders, jobs and the environment 
itself which binds all the components together. Moreover all functions of each component are defined in its respective
class and thus it is more convenient to start understanding the ENVIRONMENT class before reading the other classes
to understand the purpose of each method of each class. 

***** Comments are written above the line(s) that is being explained *****
'''


#  The machine class are the machines in some production environment.
#  Each machine contains a job list which is an ordered execution list of the jobs that are assigned to the machine.

#  The first job that comes after the lastly produced job will be moved to the "processing job" since this job is still
#  under production after some time step t. The processing job can be None if all jobs in the job list are processed
#  within the time step t or if the lastly produced job is finished after exactly time step t.

#  Each machine has a fixed conversion time which is the time that it takes for each machine to prepare from producing
#  a process p1 to a process p2 where p1 != p2. This conversion time is 0 if p1 and p2 are the same process.
class Machine(object):
    def __init__(self, machine_id, time_step=24):
        self.id = machine_id
        self.job_list = []
        self.processing_job = None
        self.finished_jobs = []
        self.conversion_time = 1.5
        self.time_step = time_step

    def append_job(self, job):
        self.job_list.append(job)

    def reset_job_list(self):
        self.job_list = []  # After time step t, the job list is reset.

    def process_jobs(self):
        self.finished_jobs = []

        if self.processing_job:  # If processing job is not None, the job is put in front of the job priority list.
            self.job_list.insert(0, self.processing_job)

        for idx, job in enumerate(self.job_list):
            # The conversion time is added if two consecutive
            #  jobs are different.
            prev_job = self.job_list[idx - 1] if idx > 0 else None
            job.process_time += self.conversion_time if prev_job and prev_job != job else 0

        #  The processing time of each job are accumulated.
        #  The jobs that are within the time step t in the accumulated job are finished.
        #  And the first job j_n which accumulated time is > t is set as the processing job
        #  if the accumulated time of the previous job j_(n-1) is < t.
        cumulative_time = np.cumsum(np.array([job.process_time for job in self.job_list]))
        finished_jobs_idx = np.argwhere(cumulative_time <= self.time_step)
        processing_job_idx = np.argmax(cumulative_time > self.time_step) \
            if (cumulative_time.size > 0 and np.max(cumulative_time) > self.time_step) else None

        self.finished_jobs = [self.job_list[int(i)] for i in finished_jobs_idx]
        self.processing_job = None

        #  If the processing job is not None, the time difference between the accumulated time of j_(n-1) and t
        #  is subtracted from the processing time of the processing job j_n since this time difference has been used
        #  on job j_n.
        if processing_job_idx:
            used_time = 0
            current_job = self.job_list[int(processing_job_idx)]
            if processing_job_idx > 0:
                previous_job = self.job_list[int(processing_job_idx) - 1]
                used_time = previous_job.process_time

            self.processing_job = current_job

            # Disabling the processing job is to avoid the job being scheduled in the following time step.
            self.processing_job.disable_job()
            current_job.process_time -= self.time_step - used_time


#  Product is a class that contains information about each product in the production catalogue of the production.
#  This includes information such as the process that is required to produce the product p, the mean quantity ordered
#  and the mean deadline associated to that particular product. This information are used to generate new orders.
class Product(object):
    def __init__(self, product_id, process_id, mean_count, mean_quantity, processes, mean_deadline=3):
        self.product_id = product_id
        self.mean_count = mean_count
        self.mean_quantity = mean_quantity
        self.mean_deadline = mean_deadline
        self.num_new_orders = 0
        self.new_orders = []
        self.process = processes[int(process_id)]

    # All random distributions regarding the quantity, deadline and the amount of new orders are assumed to be poisson
    # distributed.
    def generate_new_orders(self):
        return np.random.poisson(lam=self.mean_count)

    def generate_quantity(self):
        return np.random.poisson(lam=self.mean_quantity)

    def generate_deadline(self):
        return np.random.poisson(lam=self.mean_deadline)


#  A process is the intermediate step to produce some product.
#  This class contains information about each process such as the mean processing time and the ingoing edges (processes)
#  that must be executed before executing the process. The under processes are the ingoing edges and these are used
#  when generating a new order. Since a production does not always execute the whole production network to produce some
#  product since a production sometimes have spare parts in stock, the under processes are sampled by some probability
#  which are given in the under_processes_p.

#  Moreover, processes can only be executed at some certain machines. These are called the compatible machines.
class Process(object):
    def __init__(self, process_id, mean_process_time):
        self.process_id = process_id
        self.mean_process_time = mean_process_time
        self.under_processes = []
        self.under_processes_p = []
        self.compatible_machines = []

    #  This function is only ran in the initialization of an environment and adds the under processes to each process.
    #  Under process and its corresponding probabilities can be empty if a processes is on the top of the
    #  production network.
    def add_under_processes(self, under_processes, p):
        self.under_processes.extend(under_processes)
        self.under_processes_p.extend(p)

    #  Same as the previous function, this is only ran in the initialization of the environment.
    #  This function adds the compatible machines to each process.
    def add_compatible_machines(self, compatible_machines):
        self.compatible_machines.extend(compatible_machines)

    #  This function is used when generating new orders. The function includes an under process to the list of jobs
    #  of a new order with its corresponding inclusion probability.
    #  The function returns a list of the under processes that are included in a order which includes the process.
    def get_under_processes(self):
        process_list = []
        for process, p in zip(self.under_processes, self.under_processes_p):
            included = np.random.binomial(n=1, p=p)
            if included:
                process_list.append(process)

        return process_list

    #  The function returns the processing time of a job which is a function of the quantity that the job is assigned
    #  to produce.
    def get_processing_time(self, quantity):
        return np.random.poisson(lam=self.mean_process_time * quantity)


#  The order class contains information of an order. Each order has an ID that is unique in the current set of active
#  orders. Moreover the product and the quantity of it that has been ordered and the current deadline in whole days
#  from the current day are included in the information.
#  Moreover, the set of jobs that are associated to the order are saved as an attribute to the class. This set of jobs
#  includes the process that is the step to produce the product and the set also contains the under processes of the
#  process and their corresponding under processes.

#  Since the sequence of jobs are important, the jobs with the under processes must be finished before the job
#  containing the process can begin. Thus the leaves of the network of jobs are being enabled while the rest are
#  disabled until the leaves has been processed.
class Order(object):
    def __init__(self, order_id, product, quantity, deadline):
        self.id = order_id
        self.product = product
        self.quantity = quantity
        self.deadline = deadline
        self.jobs = []

        #  The process that produces the product is always included in the set of jobs. And thus is the root of the
        #  tree of jobs. And also is the last job that can be executed in the set of jobs.
        main_job = Job(self, product.process, quantity, deadline, None)
        self.jobs.append(main_job)

        #  Sampling the under processes to the main process.
        under_processes = product.process.get_under_processes()

        #  Creating the tree of jobs for the order where create_sub_jobs is a recursive function that returns the
        #  under processes of the main under processes.
        for under_process in under_processes:
            self.create_sub_jobs(under_process, main_job)

        #  Enabling the leaves of job tree.
        for job in self.jobs:
            if not job.under_jobs:
                job.enable_job()
            job.vectorize()

    #  This function is given a parent job and one under process of the parent process that is included in the order.
    #  This function then works recursively and creates a subtree that has the under process as the root.
    def create_sub_jobs(self, under_process, parent_job):
        sub_job = Job(self, under_process, self.quantity, self.deadline, parent_job)
        parent_job.under_jobs.append(sub_job)
        self.jobs.append(sub_job)

        sub_under_processes = under_process.get_under_processes()
        for sub_under_process in sub_under_processes:
            self.create_sub_jobs(sub_under_process, sub_job)

    #  When a job is finished, the job is removed from the set of jobs.
    def remove_job(self, job):
        self.jobs.remove(job)

    #  This function is only used in test mode when the agent(s)'s performance are measured.
    #  When several agents are working in the same environment, the newly generated orders are done in an independent
    #  environment and thus when the newly generated orders are passed into an agent's environment, the IDs must be
    #  updated. This function then is given the new ID as input and is updating the order ID of each job in the job list
    #  to the new ID.
    def update_id_jobs(self, order_id):
        for job in self.jobs:
            job.id = order_id

    #  After each time step, the deadline of the order and its jobs are decreased with the time step.
    def update_deadline(self, time_step):
        self.deadline -= time_step/24
        for job in self.jobs:
            job.update_deadline(time_step)


#  The orders class keeps track of all active orders in the environment.
#  This class makes sure that each order is given an unique ID and the amount of active orders exceeds the maximum
#  limit of orders, the surplus orders are saved in a list so these are added to the list of active orders whenever
#  the list of active orders gets reduced.
class Orders(object):
    def __init__(self):
        self.list = []
        self.surplus_orders = []
        self.current_id = 0
        self.ids_in_use = []

    # This method is used in test mode when orders are generated in an independent environment.
    # The function is given the newly generated orders and the maximum number of orders in the environment as input
    # and returns the newly generated orders with their updated IDs.
    # Also the surplus orders are put into the list of surplus orders.
    def update_ids(self, orders, max_orders):
        updated_orders = []

        for order in self.surplus_orders:
            if len(self.ids_in_use) < max_orders:
                while True:
                    if self.current_id not in self.ids_in_use:
                        break
                    else:
                        self.current_id = (self.current_id + 1) % max_orders
                order.id = self.current_id
                order.update_id_jobs(self.current_id)
                updated_orders.append(order)
                self.surplus_orders.remove(order)
                self.ids_in_use.append(self.current_id)
                self.current_id = (self.current_id + 1) % max_orders

        for order in orders:
            if len(self.ids_in_use) < max_orders:
                while True:
                    if self.current_id not in self.ids_in_use:
                        break
                    else:
                        self.current_id = (self.current_id + 1) % max_orders
                order.id = self.current_id
                order.update_id_jobs(self.current_id)
                updated_orders.append(order)
                self.ids_in_use.append(self.current_id)
                self.current_id = (self.current_id + 1) % max_orders
            else:
                self.surplus_orders.append(order)

        return updated_orders

    # When the environment is not in test mode (universal), the new orders are generated directly from here.
    # Else if universal is true, this function is used to generate the new orders which are inputted to the other
    # environments.
    def generate_new_orders(self, products, max_orders, universal=False):
        new_orders = []
        if universal: self.ids_in_use = []
        for surplus_order in self.surplus_orders:
            if len(self.ids_in_use) < max_orders:
                while True:
                    if self.current_id not in self.ids_in_use:
                        break
                    else:
                        self.current_id = (self.current_id + 1) % max_orders

            self.surplus_orders.remove(surplus_order)
            new_order = Order(self.current_id, surplus_order.product, surplus_order.quantity, surplus_order.deadline)
            new_orders.append(new_order)
            self.ids_in_use.append(self.current_id)
            self.current_id = (self.current_id + 1) % max_orders

        for product in products:
            num_new_orders = product.generate_new_orders()
            for order in range(num_new_orders):
                quantity = product.generate_quantity()
                deadline = product.generate_deadline()

                if len(self.ids_in_use) < max_orders:

                    while True:
                        if self.current_id not in self.ids_in_use:
                            break
                        else:
                            self.current_id = (self.current_id + 1) % max_orders

                    new_order = Order(self.current_id, product, quantity, deadline)
                    new_orders.append(new_order)
                    self.ids_in_use.append(self.current_id)
                    self.current_id = (self.current_id + 1) % max_orders

                else:
                    surplus_order = SurplusOrder(product, quantity, deadline)
                    self.surplus_orders.append(surplus_order)

        return new_orders

    #  When new orders are generated, they get added to the list of active orders.
    def add_orders(self, orders):
        self.list.extend(orders)

    #  When orders are finished, the IDs of the orders are removed from the active IDs and thus made available for
    #  future orders.
    #  Moreover the finished orders are removed from the list of active orders.
    def finish_orders(self, orders):
        order_ids = [order.id for order in orders]
        [self.ids_in_use.remove(o_id) for o_id in order_ids]
        [self.list.remove(order) for order in orders]

    #  The deadline of the orders that are delayed, e.g. having a deadline < 0, are saved in a list.
    #  This also includes the surplus orders since they are also active orders.
    def get_delayed_orders(self):
        delayed_orders = [np.ceil(order.deadline) for order in self.list if np.ceil(order.deadline) < 0]
        delayed_surplus_orders = [surplus_order.deadline for surplus_order in self.surplus_orders]
        return delayed_orders + delayed_surplus_orders

    #  The list of active orders can be reset if needed.
    def reset(self):
        self.list = []


#  When the number of active orders exceeds the maximum allowed number of orders, the surplus orders are created
#  which saves the generated product, quantity and deadline. Also the deadline is updated for each time step - same as
#  the orders in the active order list.
class SurplusOrder(object):
    def __init__(self, product, quantity, deadline):
        self.product = product
        self.quantity = quantity
        self.deadline = deadline

    def update_deadline(self, time_step):
        self.deadline -= time_step / 24


#  Each job is the job lists in the orders are instances of the class Job which contains information of the job.
#  This includes the order, the job is belonging to, the quantity required for the job, the deadline of the order that
#  the job is belonging to, processing time which is a hidden attribute for the environment, whether the job is enabled
#  or not, the machines that the job can be produced at and the parent job, which is for a quick enabling of the job
#  when the particular job is finished.

# Also the job has an index and a vector representation which will be explained along with the action- and state-space
class Job(object):
    def __init__(self, order, process, quantity, deadline, parent_job):
        self.order = order
        self.order_id = order.id
        self.quantity = quantity
        self.deadline = deadline
        self.ready = False
        self.process = process
        self.process_time = process.get_processing_time(self.quantity)
        self.parent_job = parent_job
        self.compatible_machines = self.process.compatible_machines
        self.compatible_machines_ids = np.array([machine.id for machine in self.compatible_machines])
        self.under_jobs = []
        self.idx = None
        self.as_vector = None

    # The jobs are updated along with the orders after each time step.
    def update_deadline(self, time_step):
        self.deadline -= time_step / 24

    # Disabling a job is used when a job is made the processing job at some machine to avoid rescheduling of the job.
    def disable_job(self):
        self.ready = False
        self.vectorize()

    # Enabling the job makes it possible for the job to be scheduled.
    # Moreover the job is re-vectorized to update the vector values of the job.
    def enable_job(self):
        self.ready = True
        self.vectorize()

    # This function vectorized the job so it can be used in the state representation of the environment.
    def vectorize(self):
        self.as_vector = np.array(
            [self.order_id, self.process.process_id, self.quantity, self.deadline, int(self.ready)])


#  The Jobs class is keeping track of all active jobs in the whole environment.
#  When new jobs are generated, these jobs get added to the list of active jobs and likewise removed when jobs
#  are finished.
class Jobs(object):
    def __init__(self):
        self.list = []

    def add_jobs(self, jobs):
        self.list.extend(jobs)

    def remove_jobs(self, jobs):
        for job in jobs:
            self.list.remove(job)
            job.order.remove_job(job)
            if job.parent_job is not None:
                job.parent_job.under_jobs.remove(job)
                if not job.parent_job.under_jobs:
                    job.parent_job.enable_job()

    # This method resets the list of jobs if needed.
    def reset(self):
        self.list = []


#  The environment class is the heart of the simulation.
#  Here the state and action spaces are defined and updated.
#  An environment is given the path to the files containing the data of machines, products and processes and
#  the maximum sizes of the environment.
class Environment(object):
    def __init__(self, environment_machines, environment_products, environment_processes,
                 min_size=15, max_jobs=50, max_orders=30, time_step=24):

        # This part reads in the data and creates the class instances of machines, products and processes.
        machines_data = pd.read_csv(environment_machines, sep=';', index_col=False)
        product_data = pd.read_csv(environment_products, sep=';', index_col=False)
        process_data = pd.read_csv(environment_processes, sep=';', index_col=False)

        self.machines = [Machine(*args) for _, args in machines_data.iterrows()]
        self.processes = [Process(*args[0:2]) for _, args in process_data.iterrows()]
        self.products = [Product(*args, self.processes) for _, args in product_data.iterrows()]

        for i, process in enumerate(self.processes):
            c_machines = process_data.iloc[i, 2][1:-1].split(',')
            c_machines = [self.machines[int(idx)] for idx in c_machines if idx != ' ']

            idx = process_data.iloc[i, 3][1:-1].split(',')
            under_processes = [self.processes[int(j)] for j in idx if j != ' ']
            under_processes_p = process_data.iloc[i, 4][1:-1].split(',')
            under_processes_p = [float(j) for j in under_processes_p if j != ' ']

            process.add_compatible_machines(c_machines)
            process.add_under_processes(under_processes=under_processes, p=under_processes_p)

        self.num_processes = len(self.processes)
        self.num_machines = len(self.machines)
        self.num_features = 5
        self.time_step = time_step
        self.jobs = Jobs()
        self.orders = Orders()
        self.t = 0
        self.max_jobs = max_jobs
        self.max_orders = max_orders
        self.action_space = np.array([])
        self.obs_actions = 1
        self.min_size = min_size
        self.num_new_jobs = []
        self.num_finished_jobs = []
        self.num_new_orders = []
        self.num_finished_orders = []

        #  The while loop generates orders so the minimum starting amount of jobs are generated.
        while len(self.jobs.list) < min_size:
            new_orders = self.orders.generate_new_orders(self.products, self.max_orders)
            self.orders.add_orders(new_orders)
            for order in new_orders:
                self.jobs.list.extend(order.jobs)

        #  Creating the current state and the action space.
        self.state = self.input_matrix()
        self.get_feasible_actions()

    #  This function returns newly generated orders. The universal variable is present so that in test mode
    #  no surplus orders are created in this environment.
    def generate_new_orders(self, universal=False):
        new_orders = self.orders.generate_new_orders(self.products, self.max_orders, universal)
        return new_orders

    #  Updating the list of active orders with the new orders. If the test mode is activated, the IDs of the new
    #  orders are updated.
    def update_order_list(self, new_orders, test=False):
        self.num_new_orders.append(len(new_orders))

        if test: new_orders = self.orders.update_ids(new_orders, self.max_orders)
        self.orders.add_orders(new_orders)
        for order in new_orders:
            self.jobs.list.extend(order.jobs)

        self.state = self.input_matrix()  # The state is updated after each intervention.
        self.get_feasible_actions()  # The action space is also updated after each intervention.

    #  Do action is the function that interacts with the environment.
    #  The function is given an action_id, which is an integer, and this action_id is used to identify
    #  which job is scheduled to which machine.
    #  If the action id is not in the action space, i.e. an invalid action, the agent is rewarded with -5, otherwise 0.

    #  If the action space is empty, the do_action function will automatically call the step function which executes
    #  the jobs in the machines and simulates a time step into the future.

    #  The function returns the new state representation, reward and whether the function has stepped into the future
    #  (new observation).
    def do_action(self, action_id, test=0):
        new_observation = False
        reward = 0

        if action_id in self.action_space:

            #  The job is identified by computing the remainder of the action ID divided by the number of machines.
            #  The machine is identified by computing the action ID modulo number of machines.
            job_id = int(action_id // self.num_machines)
            machine_id = int(action_id % self.num_machines)

            machine = self.machines[machine_id]
            job = self.jobs.list[job_id]
            machine.append_job(job)  # The job is scheduled to the machine.

            # The vector representation of the job is replaced by a 0-vector in the state representation.
            self.state[job.idx, :] = np.zeros(self.num_features)

            # All actions that regards the job that just has been scheduled are identified and removed from
            # the action space
            job_related_actions = job_id * self.num_machines + job.compatible_machines_ids
            self.action_space = self.action_space[~np.isin(self.action_space, job_related_actions)]

        elif action_id >= 0:
            reward = -5  # If illegal action, penalized with 5.

        #  If the action space is empty or if the number of actions exceeds the number of jobs, the environment
        #  steps into the future with time step t.
        if (not self.action_space.size > 0) or (self.obs_actions >= self.max_jobs):
            self.state, reward = self.step(test)
            new_observation = True
            self.obs_actions = 0

        self.obs_actions += 1

        return self.state, reward, new_observation

    #  The step function processes the jobs and generates and updates the active order- and job-list.
    #  Returns the new state and the corresponding reward.
    def step(self, test=0):
        num_finished_jobs = 0

        for machine in self.machines:
            machine.process_jobs()
            finished_jobs = machine.finished_jobs

            num_finished_jobs += len(finished_jobs)

            self.jobs.remove_jobs(finished_jobs)
            machine.reset_job_list()

        self.num_finished_jobs.append(num_finished_jobs)

        self.num_finished_orders.append(len([order for order in self.orders.list if not order.jobs]))

        #  All orders which has an empty job list are finished.
        self.orders.finish_orders([order for order in self.orders.list if not order.jobs])

        # Each deadline of orders and surplus orders are updated.
        for order in self.orders.list:
            order.update_deadline(self.time_step)

        for surplus_order in self.orders.surplus_orders:
            surplus_order.update_deadline(self.time_step)

        #  Between two steps, the reward is determined by the number of delayed orders.
        reward = self.get_reward()

        #  if the environment is not in test mode, the new orders are generated here.
        if not test:
            new_orders = self.generate_new_orders()
            self.update_order_list(new_orders)

        #  The time is increased with the time step.
        self.t += self.time_step/24

        #  If no jobs are present at time t and the environment is not in test mode
        #  the environment is stepped again into the future.
        if not self.jobs.list and not test:
            self.step()

        return self.state, reward

    #  This function is used to compute the reward between a time step.
    #  The reward is the number of delayed orders.
    def get_reward(self):
        return sum(self.orders.get_delayed_orders())

    #  This function constructs the state representation.
    #  The state representation is a n * m matrix where n is the maximum number of jobs and m is the
    #  dimension of a job vector.

    #  Each job is assigned an index idx, which corresponds to the row that the job is placed in the state matrix.
    #  When the number of active jobs is k where k < n the remaining rows in the state matrix is padded with 0-vectors.
    def input_matrix(self):
        x = np.zeros((self.max_jobs, self.num_features))
        for i, job in enumerate(self.jobs.list):
            if i < self.max_jobs:
                x[i, :] = job.as_vector
                job.idx = i
        return x

    #  Feasible actions are computed by finding the jobs that are enabled. The whole action space is thus
    #  the indices of the jobs multiplied with the number of machines and added with job's compatible machine IDs.
    def get_feasible_actions(self):
        ready_jobs = [job for job in self.jobs.list if job.ready is True]
        self.action_space = np.array([])

        for job in ready_jobs:
            if job.idx is not None:
                self.action_space = np.append(self.action_space, job.idx * self.num_machines
                                              + job.compatible_machines_ids)

    #  This function returns the actions which is prioritized by deadlines.
    #  The highest prioritized job is scheduled to the machine with the smallest job list.
    def orders_by_deadline_actions(self):
        planned_jobs = []

        for machine in self.machines:
            planned_jobs.extend(machine.job_list)

        sorted_orders = sorted(self.orders.list, key=operator.attrgetter('deadline'))
        action_space = np.array([])
        for order in sorted_orders:
            order_jobs = order.jobs
            ready_jobs = [job for job in order_jobs if job.ready and job not in planned_jobs]
            for job in ready_jobs:
                if job.idx is not None:
                    compatible_machines = job.compatible_machines
                    compatible_machines.sort(key=lambda x: len(x.job_list))
                    compatible_machines_ids = np.asarray([machine.id for machine in compatible_machines])

                    actions = job.idx * self.num_machines + compatible_machines_ids
                    action_space = np.append(action_space, actions)

        return action_space

    #  This function returns the actions which is prioritized by creation dates.
    #  The highest prioritized job is scheduled to the machine with the smallest job list.
    def orders_by_creation_actions(self):
        planned_jobs = []

        for machine in self.machines:
            planned_jobs.extend(machine.job_list)

        action_space = np.array([])
        for order in self.orders.list:
            order_jobs = order.jobs
            ready_jobs = [job for job in order_jobs if job.ready and job not in planned_jobs]
            for job in ready_jobs:
                if job.idx is not None:
                    compatible_machines = job.compatible_machines
                    compatible_machines.sort(key=lambda x: len(x.job_list))
                    compatible_machines_ids = np.asarray([machine.id for machine in compatible_machines])

                    actions = job.idx * self.num_machines + compatible_machines_ids
                    action_space = np.append(action_space, actions)
        return action_space

    #  A reset function that restarts the whole environment if needed.
    def reset(self):
        self.jobs.reset()
        self.orders.reset()
        self.t = 1

        while len(self.jobs.list) < self.min_size:
            new_orders = self.orders.generate_new_orders(self.products, self.max_orders)
            self.orders.add_orders(new_orders)
            for order in new_orders:
                self.jobs.list.extend(order.jobs)

        self.state = self.input_matrix()
        self.get_feasible_actions()