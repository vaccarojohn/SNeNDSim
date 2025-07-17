import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/NueArCC'
outfile_dir = 'graph_data'

if __name__ == "__main__":
    data_lengths = []
    data_x_start = []
    data_x_end = []
    data_y_start = []
    data_y_end = []
    data_z_start = []
    data_z_end = []
    data_in_active_volume = []
    data_in_fiducial_volume = []
    data_in_signal_volume = []
    data_outside_active_volume = []
    
    for i in range(1):
        print("Loading file " + str(i + 1) + "/1...")
        f = h5py.File(infile_dir + '/nueArCC_sns_yDir_g4_' + format(i, "04") + '.h5', 'r')

        for seg in f['segments']:
            data_lengths.append(seg['dx'])
            data_x_start.append(seg['x_start'])
            data_x_end.append(seg['x_end'])
            data_y_start.append(seg['y_start'])
            data_y_end.append(seg['y_end'])
            data_z_start.append(seg['z_start'])
            data_z_end.append(seg['z_end'])
            data_outside_active_volume.append(seg['dx'] - get_length_in_active_volume(seg['x_start'], seg['x_end'],
                                                                    seg['y_start'], seg['y_end'],
                                                                    seg['z_start'], seg['z_end']))
            data_in_active_volume.append(get_length_in_active_volume(seg['x_start'], seg['x_end'],
                                                                    seg['y_start'], seg['y_end'],
                                                                    seg['z_start'], seg['z_end']))
            data_in_fiducial_volume.append(get_length_in_fiducial_volume(seg['x_start'], seg['x_end'],
                                                                    seg['y_start'], seg['y_end'],
                                                                    seg['z_start'], seg['z_end']))
            data_in_signal_volume.append(get_length_in_signal_volume(seg['x_start'], seg['x_end'],
                                                                    seg['y_start'], seg['y_end'],
                                                                    seg['z_start'], seg['z_end']))

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/signal_segment_test_data.npz', lengths=data_lengths, x_start=data_x_start, x_end=data_x_end, y_start=data_y_start, 
                        y_end=data_y_end, z_start=data_z_start, z_end=data_z_end, outside_active_volume=data_outside_active_volume,
                        in_active_volume=data_in_active_volume, in_fiducial_volume=data_in_fiducial_volume, in_signal_volume=data_in_signal_volume)
    
    print("Data successfully written to file signal_segment_test_data.npz!")