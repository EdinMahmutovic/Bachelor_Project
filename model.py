import numpy as np
import pandas as pd
import operator


class Machine(object):
    def __init__(self, machine_id):
        self.id = machine_id
        self.job_list = []
        self.processing_job = None
        self.finished_jobs = []
        self.conversion_time = 1.5

    def append_job(self, job):
        self.job_list.append(job)

    def reset_job_list(self):
        self.job_list = []

    def process_jobs(self, time_step=1):
        self.finished_jobs = []

        if self.processing_job:
            self.job_list.insert(0, self.processing_job)
        for idx, job in enumerate(self.job_list):
            prev_job = self.job_list[idx - 1] if idx > 0 else None
            job.process_time += self.conversion_time if prev_job and prev_job != job else 0

        cumulative_time = np.cumsum(np.array([job.process_time for job in self.job_list]))
        finished_jobs_idx = np.argwhere(cumulative_time <= time_step * 24)
        processing_job_idx = np.argmax(cumulative_time > 24) if (cumulative_time.size > 0
                                                                and np.max(cumulative_time) > 24) else None

        self.finished_jobs = [self.job_list[int(i)] for i in finished_jobs_idx]
        self.processing_job = None
        if processing_job_idx:
            used_time = 0
            current_job = self.job_list[int(processing_job_idx)]
            if processing_job_idx > 0:
                previous_job = self.job_list[int(processing_job_idx) - 1]
                used_time = previous_job.process_time

            self.processing_job = current_job
            self.processing_job.disable_job()
            current_job.process_time -= 24 - used_time


class Product(object):
    def __init__(self, product_id, process_id, mean_count, mean_quantity, processes):
        self.product_id = product_id
        self.mean_count = mean_count
        self.mean_quantity = mean_quantity
        self.mean_deadline = 4
        self.num_new_orders = 0
        self.new_orders = []
        self.process = processes[int(process_id)]

    def generate_new_orders(self):
        return np.random.poisson(lam=self.mean_count)

    def generate_quantity(self):
        return np.random.poisson(lam=self.mean_quantity)

    def generate_deadline(self):
        return np.random.poisson(lam=self.mean_deadline)


class Process(object):
    def __init__(self, process_id, mean_process_time):
        self.process_id = process_id
        self.mean_process_time = mean_process_time
        self.under_processes = []
        self.under_processes_p = []
        self.compatible_machines = []

    def add_under_processes(self, under_processes, p):
        self.under_processes.extend(under_processes)
        self.under_processes_p.extend(p)

    def add_compatible_machines(self, compatible_machines):
        self.compatible_machines.extend(compatible_machines)

    def get_under_processes(self):
        process_list = []
        for process, p in zip(self.under_processes, self.under_processes_p):
            included = np.random.binomial(n=1, p=p)
            if included:
                process_list.append(process)

        return process_list

    def get_processing_time(self, quantity):
        return np.random.poisson(lam=self.mean_process_time * quantity)


class Order(object):
    def __init__(self, order_id, product, quantity, deadline):
        self.id = order_id
        self.product = product
        self.quantity = quantity
        self.deadline = deadline
        self.jobs = []

        main_job = Job(self, product.process, quantity, deadline, None)
        self.jobs.append(main_job)

        under_processes = product.process.get_under_processes()

        for under_process in under_processes:
            self.create_sub_jobs(under_process, main_job)

        for job in self.jobs:
            if not job.under_jobs:
                job.enable_job()
            job.vectorize()

    def create_sub_jobs(self, under_process, parent_job):
        sub_job = Job(self, under_process, self.quantity, self.deadline, parent_job)
        parent_job.under_jobs.append(sub_job)
        self.jobs.append(sub_job)

        sub_under_processes = under_process.get_under_processes()
        for sub_under_process in sub_under_processes:
            self.create_sub_jobs(sub_under_process, sub_job)

    def remove_job(self, job):
        self.jobs.remove(job)

    def update_deadline(self):
        self.deadline -= 1
        for job in self.jobs:
            job.update_deadline()


class SurplusOrder(object):
    def __init__(self, product, quantity, deadline):
        self.product = product
        self.quantity = quantity
        self.deadline = deadline

    def update_deadline(self):
        self.deadline -= 1


class Orders(object):
    def __init__(self):
        self.list = []
        self.surplus_orders = []
        self.current_id = 0
        self.ids_in_use = []

    def generate_new_orders(self, products, max_orders):
        new_orders = []

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

    def add_orders(self, orders):
        self.list.extend(orders)

    def finish_orders(self, orders):
        order_ids = [order.id for order in orders]
        [self.ids_in_use.remove(o_id) for o_id in order_ids]
        [self.list.remove(order) for order in orders]

    def get_delayed_orders(self):
        delayed_orders = [order.deadline for order in self.list if order.deadline < 0]
        delayed_surplus_orders = [surplus_order.deadline for surplus_order in self.surplus_orders]
        return delayed_orders + delayed_surplus_orders

    def reset(self):
        self.list = []


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

    def update_deadline(self):
        self.deadline -= 1

    def disable_job(self):
        self.ready = False
        self.vectorize()

    def enable_job(self):
        self.ready = True
        self.vectorize()

    def vectorize(self):
        self.as_vector = np.array([self.order_id, self.process.process_id, self.quantity, self.deadline, int(self.ready)])


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

    def reset(self):
        self.list = []


class Environment(object):
    def __init__(self, environment_machines, environment_products, environment_processes,
                 min_size=70, max_jobs=150, max_orders=100, time_step=1):

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

        self.num_new_jobs = []
        self.num_finished_jobs = []
        self.num_new_orders = []
        self.num_finished_orders = []

        while len(self.jobs.list) < min_size:
            new_orders = self.orders.generate_new_orders(self.products, self.max_orders)
            self.orders.add_orders(new_orders)
            for order in new_orders:
                self.jobs.list.extend(order.jobs)

        self.state = self.input_matrix()
        self.get_feasible_actions()

    def step(self):
        num_finished_jobs = 0
        for machine in self.machines:
            machine.process_jobs(self.time_step)
            finished_jobs = machine.finished_jobs

            num_finished_jobs += len(finished_jobs)

            self.jobs.remove_jobs(finished_jobs)
            machine.reset_job_list()

        self.num_finished_jobs.append(num_finished_jobs)

        self.num_finished_orders.append(len([order for order in self.orders.list if not order.jobs]))

        self.orders.finish_orders([order for order in self.orders.list if not order.jobs])
        for order in self.orders.list:
            order.update_deadline()

        for surplus_order in self.orders.surplus_orders:
            surplus_order.update_deadline()

        reward = self.get_reward()

        new_orders = self.orders.generate_new_orders(self.products, self.max_orders)
        self.num_new_orders.append(len(new_orders))

        self.orders.add_orders(new_orders)
        num_new_jobs = 0
        for order in new_orders:
            self.jobs.list.extend(order.jobs)
            num_new_jobs += len(order.jobs)

        self.num_new_jobs.append(num_new_jobs)

        self.t += self.time_step
        self.state = self.input_matrix()
        self.get_feasible_actions()

        if not self.jobs.list:
            self.step()

        return self.state, reward

    def get_reward(self):
        return sum(self.orders.get_delayed_orders())

    def input_matrix(self):
        x = np.zeros((self.max_jobs, self.num_features))
        for i, job in enumerate(self.jobs.list):
            if i < self.max_jobs:
                x[i, :] = job.as_vector
                job.idx = i
        return x

    def get_feasible_actions(self):
        ready_jobs = [job for job in self.jobs.list if job.ready is True]
        self.action_space = np.array([])

        for job in ready_jobs:
            if job.idx is not None:
                self.action_space = np.append(self.action_space, job.idx * self.num_machines
                                              + job.compatible_machines_ids)

    def do_action(self, action_id):
        new_observation = False
        reward = 0

        if action_id in self.action_space:

            job_id = int(action_id // self.num_machines)
            machine_id = int(action_id % self.num_machines)

            machine = self.machines[machine_id]
            job = self.jobs.list[job_id]
            machine.append_job(job)

            self.state[job.idx, :] = np.zeros(self.num_features)

            job_related_actions = job_id * self.num_machines + job.compatible_machines_ids
            self.action_space = self.action_space[~np.isin(self.action_space, job_related_actions)]

        elif action_id >= 0:
            reward = -1000

        if (not self.action_space.size > 0) or (self.obs_actions >= self.max_jobs):
            self.state, reward = self.step()
            new_observation = True
            self.obs_actions = 0

        self.obs_actions += 1

        return self.state, reward, new_observation

    def orders_by_deadline(self):
        sorted_orders = sorted(self.orders.list, key=operator.attrgetter('deadline'))
        return sorted_orders

    def reset(self):
        self.jobs.reset()
        self.orders.reset()
        self.t = 0
