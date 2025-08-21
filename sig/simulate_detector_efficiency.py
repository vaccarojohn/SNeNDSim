import numpy as np
from gv_significance import poisson_gaussian

NUM_SIMULATIONS = 1000
BACKGROUND_COUNTS = np.linspace(0, 400, 10)
outfile_dir = 'graph_data'

def simulate_detector_efficiency(S, B, error, threshold):
    q = 0
    total = np.random.poisson(S + B, NUM_SIMULATIONS)
    background = np.random.normal(loc=B, scale=error*B, size=NUM_SIMULATIONS)
    
    for n, b in zip(total, background):
        if poisson_gaussian.significance(n, b, 0.3*B) >= threshold:
            q += 1

    return q / NUM_SIMULATIONS

def get_counts(B, error, threshold, efficiency, S_guess=0):
    
    S = S_guess
    while simulate_detector_efficiency(S, B, error, threshold) < efficiency:
        S += 1
        
    return S

if __name__ == "__main__":
    data_3s50e = []
    data_3s90e = []
    data_3s99e = []

    S_guess_50 = 0
    S_guess_90 = 0
    S_guess_99 = 0

    print("Running simulations...")
    
    for B in BACKGROUND_COUNTS:
        if B == 0:
            data_3s50e.append(0)
            data_3s90e.append(0)
            data_3s99e.append(0)
        else:
            S_guess_50 = get_counts(B, 0.3, 3, 0.5, S_guess=S_guess_50)
            S_guess_90 = get_counts(B, 0.3, 3, 0.9, S_guess=S_guess_90)
            S_guess_99 = get_counts(B, 0.3, 3, 0.99, S_guess=S_guess_99)
            data_3s50e.append(S_guess_50)
            data_3s90e.append(S_guess_90)
            data_3s99e.append(S_guess_99)

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/sensitivity_data.npz', n3s50e=data_3s50e, n3s90e=data_3s90e, n3s99e=data_3s99e)
    print("Data successfully written to file sensitivity_data.npz!")