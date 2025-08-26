import h5py
import numpy as np
from helper_functions import get_length_in_active_volume

infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/NueArCCdirt_unprocessed'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_event_ids = []
    data_file_ids = []

    for i in range(40):
        print("Loading file " + str(i + 1) + "/40...")
        f = h5py.File(infile_dir + '/nueArCC_sns_g4_' + format(i, "04") + '.h5', 'r')
        
        event_id = f['segments'][0]['event_id']
        temp_inDet = False

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                if temp_inDet:
                    data_event_ids.append(event_id)
                    data_file_ids.append(i)
                    temp_inDet = False

                event_id = seg['event_id']
                    
            if not temp_inDet and get_length_in_active_volume(seg['x_start'], seg['x_end'], seg['y_start'], seg['y_end'], seg['z_start'], seg['z_end']) != 0:
                temp_inDet = True

        if temp_inDet:
            data_event_ids.append(event_id)
            data_file_ids.append(i)

        f.close()
            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/dirt_metadata.npz', event_ids=data_event_ids, file_ids=data_file_ids)
    
    print("Data successfully written to file dirt_metadata.npz!")