import h5py
import numpy as np
from gampixpy import config
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from helper_functions import get_length_in_fiducial_volume, get_length_in_signal_volume, get_length_in_active_volume

metadata_infile_dir = '/sdf/home/j/jvaccaro/edepsim/cosmics/graph_data'
edepsim_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/Cosmics_processed'
gampix_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/gampixpy/Cosmics'
outfile_dir = 'graph_data'
physics_config = config.default_physics_params

DB_EPSILON = 0.49
DB_MIN_SAMPLES = 6

def inverseBirksModel(dQdx):
    a = physics_config['birks_model']['birks_ab'] / physics_config['material']['w']
    b = physics_config['birks_model']['birks_kb'] / (physics_config['charge_drift']['drift_field'] * physics_config['material']['density'])
    return dQdx / (a - (b * dQdx))

if __name__ == "__main__":
    data_fenergy = []
    data_senergy = []
    data_aenergy = []
    data_pfenergy = []
    data_psenergy = []

    metadata = np.load(metadata_infile_dir + '/cosmics_metadata.npz')
    for i in range(1200):
        print("Loading file " + str(i + 1) + "/1200...")
        f = h5py.File(edepsim_infile_dir + '/CosmicFlux_g4_' + format(i, "04") + '-processed.h5', 'r')
        g = h5py.File(gampix_infile_dir + '/CosmicFlux_g4_gampixpy_' + format(i, "04") + '.h5', 'r')
        
        temp_genergy = [0, 0, 0]

        metadata_mask = metadata['file_ids'] == i
        for event_id in metadata['event_ids'][metadata_mask]:
            # Find charge readout data from GAMPixPy and recalculate `hit x` based on vertex timing
            vertex_mask = f['vertices']['event_id'] == event_id
            event_pixel_hit_mask = g['pixel_hits']['event id'] == event_id
            event_pixel_hits = g['pixel_hits'][event_pixel_hit_mask]

            if len(event_pixel_hits) == 0:
                # No GAMPixPy hits
                data_fenergy.append(0)
                data_senergy.append(0)
                data_aenergy.append(0)
                data_pfenergy.append(0)
                data_psenergy.append(0)
            else:
                drift_time = event_pixel_hits['hit t'] - f['vertices'][vertex_mask][0]['t_vert']
                tpc_index_selector = 2 * event_pixel_hits['tpc index'].astype(int) - 1 # -1 for TPC index 0, 1 for TPC index 1
                hit_x = (30 - drift_time * physics_config['charge_drift']['drift_speed']) * tpc_index_selector

                # Calculate charge deposited after recombination by accounting for electron lifetime in LAr
                charge = event_pixel_hits['hit charge'] * np.exp(drift_time / physics_config['charge_drift']['electron_lifetime'])

                # Run DBSCAN to identify hit segments
                db = DBSCAN(eps=DB_EPSILON, min_samples=DB_MIN_SAMPLES).fit(np.column_stack((hit_x, event_pixel_hits['hit y'],
                                                                                                    event_pixel_hits['hit z'])))
                segments = set(db.labels_)

                for k in segments:
                    if k == -1:
                        continue

                    segment_member_mask = db.labels_ == k
                    dQ = np.sum(charge[segment_member_mask])
                        
                    # Initialize PCA for each segment
                    pca = PCA(n_components=1)
                    transformed_data = pca.fit_transform(np.column_stack((hit_x[segment_member_mask], event_pixel_hits[segment_member_mask]['hit y'],
                                                                                                  event_pixel_hits[segment_member_mask]['hit z'])))
                        
                    # Get segment length in active, signal, and fiducial volume
                    start = np.argmin(transformed_data)
                    end = np.argmax(transformed_data)
                    dx = np.sqrt((hit_x[segment_member_mask][end] - hit_x[segment_member_mask][start])**2
                                    + (event_pixel_hits[segment_member_mask][end]['hit y'] - event_pixel_hits[segment_member_mask][start]['hit y'])**2
                                    + (event_pixel_hits[segment_member_mask][end]['hit z'] - event_pixel_hits[segment_member_mask][start]['hit z'])**2)
                        
                    in_fiducial = get_length_in_fiducial_volume(hit_x[segment_member_mask][start], hit_x[segment_member_mask][end],
                                                                event_pixel_hits[segment_member_mask][start]['hit y'],
                                                                event_pixel_hits[segment_member_mask][end]['hit y'],
                                                                event_pixel_hits[segment_member_mask][start]['hit z'],
                                                                event_pixel_hits[segment_member_mask][end]['hit z'])
                    in_signal = get_length_in_signal_volume(hit_x[segment_member_mask][start], hit_x[segment_member_mask][end],
                                                                event_pixel_hits[segment_member_mask][start]['hit y'],
                                                                event_pixel_hits[segment_member_mask][end]['hit y'],
                                                                event_pixel_hits[segment_member_mask][start]['hit z'],
                                                                event_pixel_hits[segment_member_mask][end]['hit z'])
                    
                    dEdx = inverseBirksModel(dQ / dx)
                    temp_genergy[0] += np.sum(dEdx * in_fiducial)
                    temp_genergy[1] += np.sum(dEdx * in_signal)
                    temp_genergy[2] += np.sum(dEdx * dx)
                        
                data_fenergy.append(temp_genergy[0])
                data_senergy.append(temp_genergy[1])
                data_aenergy.append(temp_genergy[2])
                data_pfenergy.append(temp_genergy[0] / temp_genergy[2])
                data_psenergy.append(temp_genergy[1] / temp_genergy[2])
                
            temp_genergy = [0, 0, 0]

        f.close()
        g.close()
            
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/cosmics_segment_data.npz', fenergy=data_fenergy, senergy=data_senergy, aenergy=data_aenergy,
                        pfenergy=data_pfenergy, psenergy=data_psenergy)
    
    print("Data successfully written to file cosmics_segment_data.npz!")