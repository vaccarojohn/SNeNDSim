import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_cosmic_ray_taggers

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/BRN'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_pmaxe = []
    data_crttop = []
    data_crtbottom = []
    data_crtleft = []
    data_crtright = []
    data_crtfront = []
    data_crtback = []
    data_crt = []
    data_light = []
    data_tmin = []
    data_tmax = []
    data_trms = []
    data_tseg = []
    data_tdiff = []
    data_pangle = []
    data_pangle2 = []

    for i in range(50):
        print("Loading file " + str(i + 1) + "/50...")
        f = h5py.File(infile_dir + '/BRN_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_protons = {}
        temp_crttop = 0
        temp_crtbottom = 0
        temp_crtleft = 0
        temp_crtright = 0
        temp_crtfront = 0
        temp_crtback = 0
        temp_tmin = 250
        temp_tmax = -50
        temp_trms = []
        temp_tseg = 0
        temp_pangle = 0
        temp_pangle2 = 0
        temp_inDet = False

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                # Save only events with energy deposition in the TPC so that data agrees with data from signal_segment_data.py
                if temp_inDet:
                    # Add energy deposited in CRT as well as minimum timestamp of energy deposition per event
                    data_crttop.append(temp_crttop)
                    data_crtbottom.append(temp_crtbottom)
                    data_crtleft.append(temp_crtleft)
                    data_crtright.append(temp_crtright)
                    data_crtfront.append(temp_crtfront)
                    data_crtback.append(temp_crtback)
                    data_crt.append(temp_crttop + temp_crtbottom + temp_crtleft + temp_crtright + temp_crtfront + temp_crtback)
                    data_light.append(temp_tmin)
                    data_tmin.append(temp_tmin)
                    data_tmax.append(temp_tmax)
                    data_pangle.append(temp_pangle)
                    data_pangle2.append(temp_pangle2)
                    
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
                temp_crttop = 0
                temp_crtbottom = 0
                temp_crtleft = 0
                temp_crtright = 0
                temp_crtfront = 0
                temp_crtback = 0
                temp_tmin = 250
                temp_tmax = -50
                temp_trms = []
                temp_tseg = 0
                temp_pangle = 0
                temp_pangle2 = 0
                temp_inDet = False
                
                event_id = seg['event_id']

            tpc_dist = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                   seg['y_start'], seg['y_end'], 
                                                   seg['z_start'], seg['z_end'])

            crt_dist = get_length_in_cosmic_ray_taggers(seg['x_start'], seg['x_end'],
                                                        seg['y_start'], seg['y_end'],
                                                        seg['z_start'], seg['z_end'])

            if tpc_dist != 0:
                temp_inDet = True
                
                # Save minimum timestamp of energy deposition segments for each event
                if seg['t0'] < temp_tmin:
                    temp_tmin = seg['t0']

                    l = np.sqrt((seg['x_end'] - seg['x_start'])**2 + (seg['y_end'] - seg['y_start'])**2)
                    if (seg['y_end'] - seg['y_start'] >= 0):
                        temp_pangle = np.arccos((seg['x_end'] - seg['x_start']) / l) if l != 0 else 0
                    else:
                        temp_pangle = np.pi - np.arccos((seg['x_end'] - seg['x_start']) / l)
                        
                    l = np.sqrt((seg['y_end'] - seg['y_start'])**2 + (seg['z_end'] - seg['z_start'])**2)
                    if (seg['z_end'] - seg['z_start'] >= 0):
                        temp_pangle2 = np.arccos((seg['y_end'] - seg['y_start']) / l) if l != 0 else 0
                    else:
                        temp_pangle2 = np.pi - np.arccos((seg['y_end'] - seg['y_start']) / l)
                
                if seg['t0'] > temp_tmax:
                    temp_tmax = seg['t0']

                temp_trms.append(seg['t0'])
                
                if (seg['t0_end'] - seg['t0_start']) > temp_tseg:
                    temp_tseg = seg['t0_end'] - seg['t0_start']

            # Save all energy deposited in cosmic ray tagger
            temp_crttop += seg['dEdx'] * crt_dist[0]
            temp_crtbottom += seg['dEdx'] * crt_dist[1]
            temp_crtleft += seg['dEdx'] * crt_dist[2]
            temp_crtright += seg['dEdx'] * crt_dist[3]
            temp_crtfront += seg['dEdx'] * crt_dist[4]
            temp_crtback += seg['dEdx'] * crt_dist[5]

            # Add all energy deposited by protons
            if seg['pdg_id'] == 2212:
                if temp_protons.get(seg['traj_id']):
                    temp_protons[seg['traj_id']] += seg['dEdx'] * tpc_dist
                else:
                    temp_protons[seg['traj_id']] = seg['dEdx'] * tpc_dist

        f.close()
            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/BRN_selection_data.npz', pmaxe=data_pmaxe, crttop=data_crttop, crtbottom=data_crtbottom, crtleft=data_crtleft,
                        crtright=data_crtright, crtfront=data_crtfront, crtback=data_crtback, crt=data_crt, light=data_light, tmin=data_tmin, tmax=data_tmax,
                        trms=data_trms, tseg=data_tseg, tdiff=data_tdiff, pangle=data_pangle, pangle2=data_pangle2)
    
    print("Data successfully written to file BRN_selection_data.npz!")