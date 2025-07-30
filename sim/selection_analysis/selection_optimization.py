import numpy as np

GRADIENT_STEP_SIZE = 0.01
OPTIMIZATION_STEP_SIZE = 0.05
OPTIMIZATION_STEP_NUM = 100
SIGNAL_WEIGHT = 116.5 / 99990
COSMICS_WEIGHT = 54
BRN_WEIGHT = 0.3192

# Read NumPy data
infile_dir = 'graph_data'
signal_segment_data = np.load('signal/' + infile_dir + '/signal_segment_data.npz')
signal_selection_data = np.load('signal/' + infile_dir + '/signal_selection_data_new.npz')
cosmics_segment_data = np.load('cosmics/' + infile_dir + '/cosmics_segment_data.npz')
cosmics_selection_data = np.load('cosmics/' + infile_dir + '/cosmics_selection_data_new.npz')
BRN_segment_data = np.load('BRN/' + infile_dir + '/BRN_segment_data.npz')
BRN_selection_data = np.load('BRN/' + infile_dir + '/BRN_selection_data_new.npz')

# Returns s / sqrt(b) with the applied selection cuts [SignalVolEFrac > c1 & SignalVolE > c2 & MaxE > c3 & TotalE < c4 & TotalE > c5 & pMaxE < 20 & MaxELength > c6 & all CRT panels < c7 & Light]
def apply_selection_cuts(c1, c2, c3, c4, c5, c6):
    # Get signal counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((signal_segment_data['psenergy'], signal_segment_data['senergy'],
                                                                               signal_segment_data['lctes'], signal_segment_data['aenergy'],
                                                                               signal_selection_data['pmaxe'], signal_segment_data['lctls'],
                                                                               signal_selection_data['crt'], signal_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > c5 and crt < c6 and light >= 0 and light <= 10:
            n += 1

    signal = n * SIGNAL_WEIGHT

    # Get cosmics counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((cosmics_segment_data['psenergy'], cosmics_segment_data['senergy'],
                                                                               cosmics_segment_data['lctes'], cosmics_segment_data['aenergy'],
                                                                               cosmics_selection_data['pmaxe'], cosmics_segment_data['lctls'],
                                                                               cosmics_selection_data['crt'], cosmics_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > c5 and crt < c6 and light >= 0 and light <= 10:
            n += 1

    background = n * COSMICS_WEIGHT

    # Get BRN counts
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((BRN_segment_data['psenergy'], BRN_segment_data['senergy'],
                                                                               BRN_segment_data['lctes'], BRN_segment_data['aenergy'],
                                                                               BRN_selection_data['pmaxe'], BRN_segment_data['lctls'],
                                                                               BRN_selection_data['crt'], BRN_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > c5 and crt < c6 and light >= 0 and light <= 10:
            n += 1

    background += n * BRN_WEIGHT

    return signal / np.sqrt(background)

# Returns a numerical approximation of the gradient of the selection cut function
def grad(c1, c2, c3, c4, c5, c6):
    dc1 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1 + GRADIENT_STEP_SIZE, c2, c3, c4, c5, c6)
                                            - apply_selection_cuts(c1 - GRADIENT_STEP_SIZE, c2, c3, c4, c5, c6))
    dc2 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2 + GRADIENT_STEP_SIZE, c3, c4, c5, c6)
                                            - apply_selection_cuts(c1, c2 - GRADIENT_STEP_SIZE, c3, c4, c5, c6))
    dc3 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3 + GRADIENT_STEP_SIZE, c4, c5, c6)
                                            - apply_selection_cuts(c1, c2, c3 - GRADIENT_STEP_SIZE, c4, c5, c6))
    dc4 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3, c4 + GRADIENT_STEP_SIZE, c5, c6)
                                            - apply_selection_cuts(c1, c2, c3, c4 - GRADIENT_STEP_SIZE, c5, c6))
    dc5 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3, c4, c5 + GRADIENT_STEP_SIZE, c6)
                                            - apply_selection_cuts(c1, c2, c3, c4, c5 - GRADIENT_STEP_SIZE, c6))
    dc6 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3, c4, c5, c6 + GRADIENT_STEP_SIZE)
                                            - apply_selection_cuts(c1, c2, c3, c4, c5, c6 - GRADIENT_STEP_SIZE))

    return np.array([dc1, dc2, dc3, dc4, dc5, dc6])

if __name__ == "__main__":
    print("Optimizing selection cuts...")
    
    maxval = apply_selection_cuts(0.10, 5, 52, 10, 1.5, 1.15)
    print("Initial params = [0.10, 5, 52, 10, 1.5, 1.15]")
    print("Initial s/sqrt(b) = " + str(maxval))
    params = np.array([0.10, 5, 52, 10, 1.5, 1.15])

    for i in range(OPTIMIZATION_STEP_NUM):
        print("Step " + str(i + 1) + "/" + str(OPTIMIZATION_STEP_NUM))
        params += (OPTIMIZATION_STEP_SIZE * grad(params[0], params[1], params[2], params[3], params[4], params[5]))

    maxval = apply_selection_cuts(params[0], params[1], params[2], params[3], params[4], params[5])
    print("Final params = " + str(params))
    print("Final s/sqrt(b) = " + str(maxval))

    
        