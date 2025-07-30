import h5py
import numpy as np
from helper_functions import get_length_in_active_volume

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/NueArCC'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_pmaxe = []
    data_crt = []
    data_light = []
    data_tmin = []
    data_tmax = []
    data_trms = []
    data_tseg = []
    data_tdiff = []

    for i in range(10):
        print("Loading file " + str(i + 1) + "/10...")
        f = h5py.File(infile_dir + '/nueArCC_sns_yDir_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_protons = {}
        temp_crt = 0
        temp_tmin = 250
        temp_tmax = -50
        temp_trms = []
        temp_tseg = 0
        temp_inDet = False

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                # Save only events with energy deposition in the TPC so that data agrees with data from signal_segment_data.py
                if temp_inDet:
                    # Add energy deposited in CRT as well as minimum timestamp of energy deposition per event
                    data_crt.append(temp_crt)
                    data_light.append(temp_tmin)
                    data_tmin.append(temp_tmin)
                    data_tmax.append(temp_tmax)
                    
                    s = 0
                    for t0 in temp_trms:
                        s += (t0 - temp_tmin)**2
                    s /= len(temp_trms)
                    data_trms.append(np.sqrt(s))
                    
                    data_tseg.append(temp_tseg)
                    data_tdiff.append(temp_tmax - temp_tmin)
                
                    # Search for proton with highest energy deposition
                    pmaxe = 0
                    for proton in temp_protons:
                        if temp_protons[proton] > pmaxe:
                            pmaxe = temp_protons[proton]
                
                    data_pmaxe.append(pmaxe)
                
                temp_protons = {}
                temp_crt = 0
                temp_tmin = 250
                temp_tmax = -50
                temp_trms = []
                temp_tseg = 0
                temp_inDet = False
                
                event_id = seg['event_id']

            tpc_dist = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                   seg['y_start'], seg['y_end'], 
                                                   seg['z_start'], seg['z_end'])

            if tpc_dist != 0:
                temp_inDet = True
                
                # Save minimum timestamp of energy deposition segments for each event
                if seg['t0'] < temp_tmin:
                    temp_tmin = seg['t0']
                
                if seg['t0'] > temp_tmax:
                    temp_tmax = seg['t0']

                temp_trms.append(seg['t0'])
                
                if (seg['t0_end'] - seg['t0_start']) > temp_tseg:
                    temp_tseg = seg['t0_end'] - seg['t0_start']

            # Because the CRT and the TPC are the only two sensitive detectors, all energy deposition segments must occur in one or the other
            temp_crt += seg['dEdx'] * (seg['dx'] - tpc_dist)

            # Add all energy deposited by protons
            if seg['pdg_id'] == 2212:
                if temp_protons.get(seg['traj_id']):
                    temp_protons[seg['traj_id']] += seg['dEdx'] * tpc_dist
                else:
                    temp_protons[seg['traj_id']] = seg['dEdx'] * tpc_dist

            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/signal_selection_data.npz', pmaxe=data_pmaxe, crt=data_crt, light=data_light, tmin=data_tmin, tmax=data_tmax, 
                        trms=data_trms, tseg=data_tseg, tdiff=data_tdiff)
    
    print("Data successfully written to file signal_selection_data.npz!")