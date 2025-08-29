import h5py
import numpy as np
from helper_functions import get_length_in_active_volume

infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/NueArCCdirt_processed'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_e01cm = []
    data_pe01cm = []
    data_e05cm = []
    data_pe05cm = []
    data_e1cm = []
    data_pe1cm = []
    data_e2cm = []
    data_pe2cm = []
    data_e3cm = []
    data_pe3cm = []

    for i in range(40):
        print("Loading file " + str(i + 1) + "/40...")
        f = h5py.File(infile_dir + '/nueArCC_sns_g4_' + format(i, "04") + '-processed.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_tmin = 250
        temp_first = 0
        temp_x = 0
        temp_y = 0
        temp_z = 0
        temp_aenergy = 0
        temp_inDet = False

        for i, seg in enumerate(f['segments']):
            if seg['event_id'] > event_id:
                # Save only events with energy deposition in the TPC so that data agrees with data from signal_segment_data.py
                if temp_inDet:
                    if temp_aenergy == 0:
                        data_e01cm.append(0)
                        data_pe01cm.append(0)
                        data_e05cm.append(0)
                        data_pe05cm.append(0)
                        data_e1cm.append(0)
                        data_pe1cm.append(0)
                        data_e2cm.append(0)
                        data_pe2cm.append(0)
                        data_e3cm.append(0)
                        data_pe3cm.append(0)
                    else:
                        # Initialize temporary energy counters
                        temp_e01cm = 0
                        temp_e05cm = 0
                        temp_e1cm = 0
                        temp_e2cm = 0
                        temp_e3cm = 0

                        # Find all energy deposition sufficiently close to the first segment in the TPC
                        for seg2 in f['segments'][temp_first:i]:
                            d = np.sqrt(((seg2['x_start'] + seg2['x_end']) / 2 - temp_x)**2 + ((seg2['y_start'] + seg2['y_end']) / 2 - temp_y)**2
                                        + ((seg2['z_start'] + seg2['z_end']) / 2 - temp_z)**2)

                            energy = seg2['dEdx'] * get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                                                seg['y_start'], seg['y_end'], 
                                                                                seg['z_start'], seg['z_end'])
                        
                            if d < 0.1:
                                temp_e01cm += energy
                            if d < 0.5:
                                temp_e05cm += energy
                            if d < 1:
                                temp_e1cm += energy
                            if d < 2:
                                temp_e2cm += energy
                            if d < 3:
                                temp_e3cm += energy

                        data_e01cm.append(temp_e01cm)
                        data_pe01cm.append(temp_e01cm / temp_aenergy)
                        data_e05cm.append(temp_e05cm)
                        data_pe05cm.append(temp_e05cm / temp_aenergy)
                        data_e1cm.append(temp_e1cm)
                        data_pe1cm.append(temp_e1cm / temp_aenergy)
                        data_e2cm.append(temp_e2cm)
                        data_pe2cm.append(temp_e2cm / temp_aenergy)
                        data_e3cm.append(temp_e3cm)
                        data_pe3cm.append(temp_e3cm / temp_aenergy)
                    
                temp_tmin = 250
                temp_first = i
                temp_x = 0
                temp_y = 0
                temp_z = 0
                temp_aenergy = 0
                temp_inDet = False
                
                event_id = seg['event_id']

            tpc_dist = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                   seg['y_start'], seg['y_end'], 
                                                   seg['z_start'], seg['z_end'])
            if tpc_dist != 0:
                temp_inDet = True
                temp_aenergy += seg['dEdx'] * tpc_dist
                
                # Save minimum timestamp of energy deposition segments for each event
                if seg['t0'] < temp_tmin:
                    temp_tmin = seg['t0']
                    temp_x = (seg['x_start'] + seg['x_end']) / 2
                    temp_y = (seg['y_start'] + seg['y_end']) / 2
                    temp_z = (seg['z_start'] + seg['z_end']) / 2

        # Process final event in file
        if temp_inDet:
            if temp_aenergy == 0:
                data_e01cm.append(0)
                data_pe01cm.append(0)
                data_e05cm.append(0)
                data_pe05cm.append(0)
                data_e1cm.append(0)
                data_pe1cm.append(0)
                data_e2cm.append(0)
                data_pe2cm.append(0)
                data_e3cm.append(0)
                data_pe3cm.append(0)
            else:
                # Initialize temporary energy counters
                temp_e01cm = 0
                temp_e05cm = 0
                temp_e1cm = 0
                temp_e2cm = 0
                temp_e3cm = 0

                # Find all energy deposition sufficiently close to the first segment in the TPC
                for seg2 in f['segments'][temp_first:i]:
                    d = np.sqrt(((seg2['x_start'] + seg2['x_end']) / 2 - temp_x)**2 + ((seg2['y_start'] + seg2['y_end']) / 2 - temp_y)**2
                                + ((seg2['z_start'] + seg2['z_end']) / 2 - temp_z)**2)

                    energy = seg2['dEdx'] * get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                                        seg['y_start'], seg['y_end'], 
                                                                        seg['z_start'], seg['z_end'])
                    
                    if d < 0.1:
                        temp_e01cm += energy
                    if d < 0.5:
                        temp_e05cm += energy
                    if d < 1:
                        temp_e1cm += energy
                    if d < 2:
                        temp_e2cm += energy
                    if d < 3:
                        temp_e3cm += energy

                data_e01cm.append(temp_e01cm)
                data_pe01cm.append(temp_e01cm / temp_aenergy)
                data_e05cm.append(temp_e05cm)
                data_pe05cm.append(temp_e05cm / temp_aenergy)
                data_e1cm.append(temp_e1cm)
                data_pe1cm.append(temp_e1cm / temp_aenergy)
                data_e2cm.append(temp_e2cm)
                data_pe2cm.append(temp_e2cm / temp_aenergy)
                data_e3cm.append(temp_e3cm)
                data_pe3cm.append(temp_e3cm / temp_aenergy)

        f.close()
            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/dirt_vertex_data.npz', e01cm=data_e01cm, pe01cm=data_pe01cm, e05cm=data_e05cm, pe05cm=data_pe05cm, e1cm=data_e1cm,
                        pe1cm=data_pe1cm, e2cm=data_e2cm, pe2cm=data_pe2cm, e3cm=data_e3cm, pe3cm=data_pe3cm)
    
    print("Data successfully written to file dirt_vertex_data.npz!")