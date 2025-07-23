import h5py
import numpy as np
from helper_functions import get_length_in_active_volume

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/Cosmics'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_pmaxe = []
    data_crt = []
    data_light = []

    for i in range(200):
        print("Loading file " + str(i + 1) + "/200...")
        f = h5py.File(infile_dir + '/CosmicFlux_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_protons = {}
        temp_crt = 0
        temp_t0 = 200
        temp_inDet = False

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                # Save only events with energy deposition in the TPC so that data agrees with data from signal_segment_data.py
                if temp_inDet:
                    # Add energy deposited in CRT as well as minimum timestamp of energy deposition per event
                    data_crt.append(temp_crt)
                    data_light.append(temp_t0)
                
                    # Search for proton with highest energy deposition
                    pmaxe = 0
                    for proton in temp_protons:
                        if temp_protons[proton] > pmaxe:
                            pmaxe = temp_protons[proton]
                
                    data_pmaxe.append(pmaxe)
                
                temp_protons = {}
                temp_crt = 0
                temp_t0 = 200
                temp_inDet = False
                
                event_id = seg['event_id']

            tpc_dist = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                   seg['y_start'], seg['y_end'], 
                                                   seg['z_start'], seg['z_end'])

            if tpc_dist != 0:
                temp_inDet = True

            # Because the CRT and the TPC are the only two sensitive detectors, all energy deposition segments must occur in one or the other
            temp_crt += seg['dEdx'] * (seg['dx'] - tpc_dist)

            # Save minimum timestamp of energy deposition segments for each event
            if tpc_dist != seg['dx'] and seg['t0'] < temp_t0:
                temp_t0 = seg['t0']

            # Add all energy deposited by protons
            if seg['pdg_id'] == 2212:
                if temp_protons.get(seg['traj_id']):
                    temp_protons[seg['traj_id']] += seg['dE']
                else:
                    temp_protons[seg['traj_id']] = seg['dE']

            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/cosmics_selection_data.npz', pmaxe=data_pmaxe, crt=data_crt, light=data_light)
    
    print("Data successfully written to file cosmics_selection_data.npz!")