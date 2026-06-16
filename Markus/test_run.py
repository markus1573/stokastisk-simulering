import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

# --- Task 1: Setup and Basic Simulation ---
# Transition matrix P (1 time step = 1 month)
P = np.array([
    [0.9915, 0.005,  0.0025, 0,      0.001 ],
    [0,      0.986,  0.005,  0.004,  0.005 ],
    [0,      0,      0.992,  0.003,  0.005 ],
    [0,      0,      0,      0.991,  0.009 ],
    [0,      0,      0,      0,      1.0   ]
])

def simulate_woman(P):
    """Simulates a single woman's path until death (state 5, index 4)."""
    state = 0
    path = [state]
    while state != 4:
        state = np.random.choice(5, p=P[state])
        path.append(state)
    return path

np.random.seed(42)
n_sims = 1000
lifetimes = []
local_recurrence_count = 0
states_at_120 = []

for _ in range(n_sims):
    path = simulate_woman(P)
    lifetimes.append(len(path) - 1) # Lifetime = number of transitions
    
    # Check for local recurrence: state 2 or state 4 (index 1 or 3)
    if 1 in path or 3 in path:
        local_recurrence_count += 1
        
    # State at t = 120
    if len(path) - 1 >= 120:
        states_at_120.append(path[120])
    else:
        states_at_120.append(4) # Already absorbed in death state

# Simulated proportions
states = np.arange(1, 6) # States 1 to 5 (indices 0 to 4)
sim_proportions = np.array([states_at_120.count(i) / n_sims for i in range(5)])

# Theoretical proportions: P^120 starting from state 1 (index 0)
u0 = np.array([1, 0, 0, 0, 0])
theo_proportions = matrix_power(P, 120)[0]

print("Simulated proportions:", sim_proportions)
print("Theoretical proportions:", theo_proportions)
