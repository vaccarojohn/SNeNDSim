import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/BRN'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_echanges = []
    data_lengths = []
    data_palengths = []
    data_pslengths = []
    data_pflengths = []

    for i in range(50):
        print("Loading file " + str(i + 1) + "/10...")
        f = h5py.File(infile_dir + '/BRN_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['trajectories'][0]['event_id']
        
        for traj in f['trajectories']:
            l = np.sqrt(np.sum(np.square(traj['xyz_start'] - traj['xyz_end'])))
            
            if l == 0:
                continue

            data_echanges.append(traj['E_start'] - traj['E_end'])
            data_lengths.append(l)
            data_palengths.append(get_length_in_active_volume(traj['xyz_start'][0], traj['xyz_end'][0], traj['xyz_start'][1], 
                                                             traj['xyz_end'][1], traj['xyz_start'][2], traj['xyz_end'][2]) / l)
            data_pslengths.append(get_length_in_signal_volume(traj['xyz_start'][0], traj['xyz_end'][0], traj['xyz_start'][1], 
                                                             traj['xyz_end'][1], traj['xyz_start'][2], traj['xyz_end'][2]) / l)
            data_pflengths.append(get_length_in_fiducial_volume(traj['xyz_start'][0], traj['xyz_end'][0], traj['xyz_start'][1], 
                                                               traj['xyz_end'][1], traj['xyz_start'][2], traj['xyz_end'][2]) / l)
            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/BRN_trajectory_data.npz', echanges=data_echanges, lengths=data_lengths, palengths=data_palengths, 
                                                                     pslengths=data_pslengths, pflengths=data_pflengths)
    
    print("Data successfully written to file BRN_trajectory_data.npz!")