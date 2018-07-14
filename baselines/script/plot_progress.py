import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.stats as sts

delta = .05

avg_return = []
bound = []
var = []
max_iw = []
d_4 = []
kl = []
with open('../results/trpo/mlp/progress.csv') as fp:
    reader = csv.DictReader(fp)
    for row in  reader:
        kl.append(min(np.infty, float(row['meankl'])))
        avg_return.append(min(100,float(row['J_hat'])))
        bound.append(float(row['Our_bound']))
        var.append(min(1e6,float(row['Var_J'])))
        max_iw.append(min(np.infty, float(row['Max_iw'])))
        d_4.append(min(np.infty, float(row['Reny_4'])))
iterations = len(avg_return)

plt.plot(range(iterations), kl)
#plt.plot(range(iterations), avg_return)
#plt.plot(range(len(bound)), np.clip(bound, -100, np.infty))
#plt.plot(range(iterations), var)
#plt.plot(range(iterations), max_iw)
#plt.plot(range(iterations), d_4)
plt.show()