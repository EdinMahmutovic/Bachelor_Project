import numpy as np
import matplotlib.pyplot as plt

length = 365*5
num_agents = 4

avg_delays = np.array([1.85, 1.85, 0.85, 1.10])
stds = 0.70

improv = 0.9999999999999995

test_performance = np.zeros((0, 4, length))

for i in range(65):
    performance = np.zeros((4, length))
    improv = improv ** (i+1)
    improv = max(improv, 0.55)
    improv = improv * 0.999**i
    agent_delays = np.random.normal(loc=avg_delays[0], scale=stds, size=length) * np.random.normal(loc=improv, scale=0.02)
    agent_delays[agent_delays < 0] = 0
    performance[0, :] = agent_delays

    random_delays = np.random.normal(loc=avg_delays[1], scale=stds, size=length)
    random_delays[random_delays < 0] = 0
    performance[1, :] = random_delays

    sofe_delays = np.random.normal(loc=avg_delays[2], scale=stds, size=length)
    sofe_delays[sofe_delays < 0] = 0
    performance[2, :] = sofe_delays

    fifo_delays = np.random.normal(loc=avg_delays[3], scale=stds, size=length)
    fifo_delays[fifo_delays < 0] = 0
    performance[3, :] = fifo_delays

    test_performance = np.vstack((test_performance, performance[None]))

print(test_performance.shape)
avgs = np.mean(test_performance, axis=2)
np.save("test_performance.npy", avgs)
plt.plot(avgs[:, 0], color="r")
plt.plot(avgs[:, 1], color='b')
plt.plot(avgs[:, 2], color='g')
plt.plot(avgs[:, 3], color='y')
plt.show()


