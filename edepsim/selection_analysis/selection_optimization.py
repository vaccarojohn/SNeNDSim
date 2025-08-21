import numpy as np

GRADIENT_STEP_SIZE = 0.01
OPTIMIZATION_STEP_SIZE = 0.01
OPTIMIZATION_STEP_NUM = 100
SIGNAL_WEIGHT = 116.5 / 99990
COSMICS_WEIGHT = 54
BRN_WEIGHT = 0.3192

# Read NumPy data
infile_dir = 'graph_data'
signal_segment_data = np.load('../signal/' + infile_dir + '/signal_segment_data_reduced.npz')
signal_selection_data = np.load('../signal/' + infile_dir + '/signal_selection_data_reduced.npz')
cosmics_segment_data = np.load('../cosmics/' + infile_dir + '/cosmics_segment_data_reduced.npz')
cosmics_selection_data = np.load('../cosmics/' + infile_dir + '/cosmics_selection_data_reduced.npz')
BRN_segment_data = np.load('../BRN/' + infile_dir + '/BRN_segment_data_reduced.npz')
BRN_selection_data = np.load('../BRN/' + infile_dir + '/BRN_selection_data_reduced.npz')

# Returns s / sqrt(b) with the applied selection cuts [SignalVolEFrac > c1 & SignalVolE > 5 & MaxE > c2 & TotalE < c3 & TotalE > c4 & pMaxE < 20 & all CRT panels < c5 & Light]
def apply_selection_cuts(c1, c2, c3, c4, c5):
    # Get signal counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((signal_segment_data['psenergy'], signal_segment_data['senergy'],
                                                                               signal_segment_data['lctes'], signal_segment_data['aenergy'],
                                                                               signal_selection_data['pmaxe'], signal_segment_data['lctls'],
                                                                               signal_selection_data['crt'], signal_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > 2 and crt < c5 and light >= 0 and light <= 10:
            n += 1

    signal = n * SIGNAL_WEIGHT

    # Get cosmics counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((cosmics_segment_data['psenergy'], cosmics_segment_data['senergy'],
                                                                               cosmics_segment_data['lctes'], cosmics_segment_data['aenergy'],
                                                                               cosmics_selection_data['pmaxe'], cosmics_segment_data['lctls'],
                                                                               cosmics_selection_data['crt'], cosmics_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > 2 and crt < c5 and light >= 0 and light <= 10:
            n += 1

    background = n * COSMICS_WEIGHT

    # Get BRN counts
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, light in np.column_stack((BRN_segment_data['psenergy'], BRN_segment_data['senergy'],
                                                                               BRN_segment_data['lctes'], BRN_segment_data['aenergy'],
                                                                               BRN_selection_data['pmaxe'], BRN_segment_data['lctls'],
                                                                               BRN_selection_data['crt'], BRN_selection_data['light'])):
        
        if psenergy > c1 and senergy > 5 and lcte > c2 and aenergy < c3 and aenergy > c4 and pmaxe < 20 and lctl > 2 and crt < c5 and light >= 0 and light <= 10:
            n += 1

    background += n * BRN_WEIGHT

    return signal / np.sqrt(background)

# Returns a numerical approximation of the gradient of the selection cut function
def grad(c1, c2, c3, c4, c5):
    dc1 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1 + GRADIENT_STEP_SIZE, c2, c3, c4, c5)
                                            - apply_selection_cuts(c1 - GRADIENT_STEP_SIZE, c2, c3, c4, c5))
    dc2 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2 + GRADIENT_STEP_SIZE, c3, c4, c5)
                                            - apply_selection_cuts(c1, c2 - GRADIENT_STEP_SIZE, c3, c4, c5))
    dc3 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3 + GRADIENT_STEP_SIZE, c4, c5)
                                            - apply_selection_cuts(c1, c2, c3 - GRADIENT_STEP_SIZE, c4, c5))
    dc4 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3, c4 + GRADIENT_STEP_SIZE, c5)
                                            - apply_selection_cuts(c1, c2, c3, c4 - GRADIENT_STEP_SIZE, c5))
    dc5 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2, c3, c4, c5 + GRADIENT_STEP_SIZE)
                                            - apply_selection_cuts(c1, c2, c3, c4, c5 - GRADIENT_STEP_SIZE))

    return np.array([dc1, dc2, dc3, dc4, dc5])

if __name__ == "__main__":
    print("Optimizing selection cuts...")
    
    maxval = apply_selection_cuts(0.1, 5.01, 52, 9.99, 1.16)
    print("Initial params = [0.1, 5.01, 52, 9.99, 1.16]")
    print("Initial s/sqrt(b) = " + str(maxval))
    params = np.array([0.1, 5.01, 52, 9.99, 1.16])

    for i in range(OPTIMIZATION_STEP_NUM):
        print("Step " + str(i + 1) + "/" + str(OPTIMIZATION_STEP_NUM))
        params += (OPTIMIZATION_STEP_SIZE * grad(params[0], params[1], params[2], params[3], params[4]))

    maxval = apply_selection_cuts(params[0], params[1], params[2], params[3], params[4])
    print("Final params = " + str(params))
    print("Final s/sqrt(b) = " + str(maxval))

    
        