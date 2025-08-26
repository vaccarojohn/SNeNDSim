import numpy as np

GRADIENT_STEP_SIZE = 0.01
OPTIMIZATION_STEP_SIZE = 0.01
OPTIMIZATION_STEP_NUM = 10
SIGNAL_WEIGHT = 116.5 / 100000
COSMICS_WEIGHT = 9
BRN_WEIGHT = 0.312
DIRT_WEIGHT = 0.000233

# Read NumPy data
infile_dir = 'graph_data'
signal_segment_data = np.load('../signal/' + infile_dir + '/signal_segment_data.npz')
signal_selection_data = np.load('../signal/' + infile_dir + '/signal_selection_data.npz')
cosmics_segment_data = np.load('../cosmics/' + infile_dir + '/cosmics_segment_data.npz')
cosmics_selection_data = np.load('../cosmics/' + infile_dir + '/cosmics_selection_data.npz')
BRN_segment_data = np.load('../BRN/' + infile_dir + '/BRN_segment_data.npz')
BRN_selection_data = np.load('../BRN/' + infile_dir + '/BRN_selection_data.npz')
dirt_segment_data = np.load('../dirt/' + infile_dir + '/dirt_segment_data.npz')
dirt_selection_data = np.load('../dirt/' + infile_dir + '/dirt_selection_data.npz')

# Returns s / sqrt(b) with the applied selection cuts [SignalVolEFrac > 0.1 & SignalVolE > 5 & MaxE > 5 & TotalE < 52 & TotalE > 10 & pMaxE < 20 & all CRT panels < 1.2 & tMin > c1 & tMin < c2]
def apply_selection_cuts(c1, c2):
    # Get signal counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, tmin in np.column_stack((signal_segment_data['psenergy'], signal_segment_data['senergy'],
                                                                               signal_segment_data['lctes'], signal_segment_data['aenergy'],
                                                                               signal_selection_data['pmaxe'], signal_segment_data['lctls'],
                                                                               signal_selection_data['crt'], signal_selection_data['tmin'])):
        
        if psenergy > 0.1 and senergy > 5 and lcte > 5 and aenergy < 52 and aenergy > 10 and pmaxe < 20 and lctl > 2 and crt < 1.2 and tmin > c1 and tmin < c2:
            n += 1

    signal = n * SIGNAL_WEIGHT

    # Get cosmics counts
    n = 0
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, tmin in np.column_stack((cosmics_segment_data['psenergy'], cosmics_segment_data['senergy'],
                                                                               cosmics_segment_data['lctes'], cosmics_segment_data['aenergy'],
                                                                               cosmics_selection_data['pmaxe'], cosmics_segment_data['lctls'],
                                                                               cosmics_selection_data['crt'], cosmics_selection_data['tmin'])):
        
        if psenergy > 0.1 and senergy > 5 and lcte > 5 and aenergy < 52 and aenergy > 10 and pmaxe < 20 and lctl > 2 and crt < 1.2 and tmin > c1 and tmin < c2:
            n += 1

    background = n * COSMICS_WEIGHT

    # Get BRN counts
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, tmin in np.column_stack((BRN_segment_data['psenergy'], BRN_segment_data['senergy'],
                                                                               BRN_segment_data['lctes'], BRN_segment_data['aenergy'],
                                                                               BRN_selection_data['pmaxe'], BRN_segment_data['lctls'],
                                                                               BRN_selection_data['crt'], BRN_selection_data['tmin'])):
        
        if psenergy > 0.1 and senergy > 5 and lcte > 5 and aenergy < 52 and aenergy > 10 and pmaxe < 20 and lctl > 2 and crt < 1.2 and tmin > c1 and tmin < c2:
            n += 1

    background += n * BRN_WEIGHT

    # Get dirt counts
    for psenergy, senergy, lcte, aenergy, pmaxe, lctl, crt, tmin in np.column_stack((dirt_segment_data['psenergy'], dirt_segment_data['senergy'],
                                                                               dirt_segment_data['lctes'], dirt_segment_data['aenergy'],
                                                                               dirt_selection_data['pmaxe'], dirt_segment_data['lctls'],
                                                                               dirt_selection_data['crt'], dirt_selection_data['tmin'])):
        
        if psenergy > 0.1 and senergy > 5 and lcte > 5 and aenergy < 52 and aenergy > 10 and pmaxe < 20 and lctl > 2 and crt < 1.2 and tmin > c1 and tmin < c2:
            n += 1

    background += n * DIRT_WEIGHT

    return signal / np.sqrt(background)

# Returns a numerical approximation of the gradient of the selection cut function
def grad(c1, c2):
    dc1 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1 + GRADIENT_STEP_SIZE, c2)
                                            - apply_selection_cuts(c1 - GRADIENT_STEP_SIZE, c2))
    dc2 = (1 / (2 * GRADIENT_STEP_SIZE)) * (apply_selection_cuts(c1, c2 + GRADIENT_STEP_SIZE)
                                            - apply_selection_cuts(c1, c2 - GRADIENT_STEP_SIZE))


    return np.array([dc1, dc2])

if __name__ == "__main__":
    print("Optimizing selection cuts...")
    
    maxval = apply_selection_cuts(0.25, 8)
    print("Initial params = [0.25, 8]")
    print("Initial s/sqrt(b) = " + str(maxval))
    params = np.array([0.25, 8])

    for i in range(OPTIMIZATION_STEP_NUM):
        print("Step " + str(i + 1) + "/" + str(OPTIMIZATION_STEP_NUM))
        params += (OPTIMIZATION_STEP_SIZE * grad(params[0], params[1]))

    maxval = apply_selection_cuts(params[0], params[1])
    print("Final params = " + str(params))
    print("Final s/sqrt(b) = " + str(maxval))

    
        