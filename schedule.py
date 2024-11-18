import cvxpy as cp
import numpy as np
import pandas as pd

#Setting up array of runtimes in minutes
songs = pd.read_csv('Path to csv')
songs_with_runtime = songs[['album', 'name','duration_ms']]
songs_with_runtime = songs_with_runtime.head(60)
runtime = songs_with_runtime['duration_ms'].to_numpy() / 60000

setlist = {0,1,2,3}


x = cp.Variable((len(runtime),7), boolean = True)
y = cp.Variable((len(runtime),7), boolean = True)

z_1 = cp.Variable(len(runtime), boolean = True)
z_2 = cp.Variable(len(runtime), boolean = True)
w_1 = cp.Variable(len(runtime), boolean = True)
w_2 = cp.Variable(len(runtime), boolean = True)

d= cp.Variable()
s = cp.Variable()

constraints = []

# Total listening time for Alex and Blake across all days
T_A = cp.sum(cp.multiply(runtime[:, None], x))
T_B = cp.sum(cp.multiply(runtime[:, None], y))

constraints += [T_A - T_B <= 1000 * z_1, T_B - T_A <= 1000 * z_2]
constraints += [d >= T_A - T_B - 1000 * (1 - z_1), d >= T_B - T_A - 1000 * (1 - z_2)]
constraints += [T_A >= T_B]

for i in range(len(runtime)):
    if i in setlist:
        constraints.append(cp.sum(x[i,:]) == 1)
        constraints.append(cp.sum(y[i,:]) == 1)
    else:
        constraints.append(cp.sum(x[i, :]) + cp.sum(y[i, :]) == 1)

S_A = cp.sum(x)  # Total songs Alex listens to
S_B = cp.sum(y)  # Total songs Blake listens to
constraints += [S_A - S_B <= 100 * w_1, S_B - S_A <= 100 * w_2]
constraints += [s >= S_A - S_B - 100 * (1 - w_1), s >= S_B - S_A - 100 * (1 - w_2)]

for d in range(7):
    constraints.append(cp.sum(cp.multiply(runtime, x[:, d])) <= 30)
    constraints.append(cp.sum(cp.multiply(runtime, y[:, d])) <= 30)

problem = cp.Problem(cp.Minimize(0), constraints)

problem.solve(solver=cp.GUROBI, verbose = True)
print(x.value)
print(y.value)
