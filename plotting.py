import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('num_jobs.pkl', 'rb') as f:
    num_jobs = pickle.load(f)

with open('avg_delays7.pkl', 'rb') as f:
    avg_7 = pickle.load(f)

with open('avg_delays180.pkl', 'rb') as f:
    avg_180 = pickle.load(f)


#plt.plot(num_jobs, color="r")
#plt.plot(avg_7, color="b")
plt.plot(avg_7, color='g')
plt.show()
