import h5py
import numpy as np
from gampixpy import config
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from helper_functions import get_theta, edge_dist, get_length_in_fiducial_volume, get_length_in_signal_volume, get_length_in_active_volume

metadata_infile_dir = '/sdf/home/j/jvaccaro/SNeNDSim/edepsim/BRN/graph_data'
edepsim_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/BRN'
gampix_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/gampixpy/BRN'
outfile_dir = 'graph_data'
physics_config = config.default_physics_params
readout_config = config.preset_readout_configs['GAMPixD_notruth']
waveform_ones = np.ones(20) # readout_config['pixels']['integration_length']
waveform_timesteps = np.arange(0, 20) * readout_config['pixels']['clock_interval'] # readout_config['pixels']['integration_length']

DB_EPSILON = 0.7
DB_MIN_SAMPLES = 3

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
    data_lctes = []
    data_lctls = []
    data_closestedge = []
    data_theta = []

    metadata = np.load(metadata_infile_dir + '/BRN_metadata.npz')
    for i in range(80):
        print("Loading file " + str(i + 1) + "/80...")
        f = h5py.File(edepsim_infile_dir + '/BRN_g4_' + format(i, "04") + '.h5', 'r')
        g = h5py.File(gampix_infile_dir + '/BRN_g4_gampixpy_' + format(i, "04") + '.h5', 'r')

        metadata_mask = metadata['file_ids'] == i
        for event_id in metadata['event_ids'][metadata_mask]:
            # Find charge readout data from GAMPixPy and recalculate `hit x` based on vertex timing
            segments_mask = f['segments']['event_id'] == event_id
            available_segments = f['segments'][segments_mask]
            event_pixel_hit_mask = g['pixels']['event id'] == event_id
            event_pixel_hits = g['pixels'][event_pixel_hit_mask]
            available_segment_indices = np.array([np.where(available_segments['segment_id'] == label)[0][0] for label in event_pixel_hits['label'][:,0]])

            if len(event_pixel_hits) == 0:
                # No GAMPixPy hits
                data_fenergy.append(0)
                data_senergy.append(0)
                data_aenergy.append(0)
                data_pfenergy.append(0)
                data_psenergy.append(0)
                data_lctes.append(0)
                data_lctls.append(0)
                data_closestedge.append(0)
                data_theta.append(0)
            else:
                drift_time = np.add.outer(event_pixel_hits['trig t'] - available_segments[available_segment_indices]['t0_start'], waveform_timesteps)
                tpc_index_selector = 2 * event_pixel_hits['pixel tpc'].astype(int) - 1 # -1 for TPC index 0, 1 for TPC index 1
                hit_x = np.einsum('ij,i->ij', (30 - drift_time * physics_config['charge_drift']['drift_speed']), tpc_index_selector).flatten()
                hit_y = event_pixel_hits['ext pos'][:,:,1].flatten()
                hit_z = event_pixel_hits['ext pos'][:,:,2].flatten()

                # Calculate charge deposited after recombination by accounting for electron lifetime in LAr
                charge = (event_pixel_hits['waveform'] * np.exp(drift_time / physics_config['charge_drift']['electron_lifetime'])).flatten()

                # Run DBSCAN to identify hit segments
                db = DBSCAN(eps=DB_EPSILON, min_samples=DB_MIN_SAMPLES).fit(np.column_stack((hit_x, hit_y, hit_z)))
                trajectories = set(db.labels_)

                maximum_energy = -1
                maximum_energy_length = -1
                closest_edge = 0
                theta = 0
                temp_genergy = [0, 0, 0]
                for traj in trajectories:
                    if traj == -1:
                        continue

                    traj_member_mask = db.labels_ == traj
                    dQ = np.sum(charge[traj_member_mask])
                        
                    # Initialize PCA for each segment
                    pca = PCA(n_components=1)
                    transformed_data = pca.fit_transform(np.column_stack((hit_x[traj_member_mask], hit_y[traj_member_mask],
                                                                                                   hit_z[traj_member_mask])))
                        
                    # Get segment length in active, signal, and fiducial volume
                    start = np.argmin(transformed_data)
                    end = np.argmax(transformed_data)
                    dx = np.sqrt((hit_x[traj_member_mask][end] - hit_x[traj_member_mask][start])**2
                                    + (hit_y[traj_member_mask][end] - hit_y[traj_member_mask][start])**2
                                    + (hit_z[traj_member_mask][end] - hit_z[traj_member_mask][start])**2)
                        
                    in_fiducial = get_length_in_fiducial_volume(hit_x[traj_member_mask][start], 
                                                                hit_x[traj_member_mask][end],
                                                                hit_y[traj_member_mask][start],
                                                                hit_y[traj_member_mask][end],
                                                                hit_z[traj_member_mask][start],
                                                                hit_z[traj_member_mask][end])
                    in_signal = get_length_in_signal_volume(hit_x[traj_member_mask][start],
                                                            hit_x[traj_member_mask][end],
                                                            hit_y[traj_member_mask][start],
                                                            hit_y[traj_member_mask][end],
                                                            hit_z[traj_member_mask][start],
                                                            hit_z[traj_member_mask][end])
                    
                    dEdx = inverseBirksModel(dQ / dx)
                    dE = dEdx * dx
                    if dE > maximum_energy:
                        maximum_energy = dE
                        maximum_energy_length = dx
                        closest_edge = edge_dist(hit_x[traj_member_mask][start], 
                                                 hit_x[traj_member_mask][end],
                                                 hit_y[traj_member_mask][start],
                                                 hit_y[traj_member_mask][end],
                                                 hit_z[traj_member_mask][start],
                                                 hit_z[traj_member_mask][end],
                                                 30, 25, 30)
                        
                        theta = get_theta(hit_x[traj_member_mask][start], 
                                          hit_x[traj_member_mask][end],
                                          hit_y[traj_member_mask][start],
                                          hit_y[traj_member_mask][end],
                                          hit_z[traj_member_mask][start],
                                          hit_z[traj_member_mask][end])
                    
                    temp_genergy[0] += np.sum(dEdx * in_fiducial)
                    temp_genergy[1] += np.sum(dEdx * in_signal)
                    temp_genergy[2] += np.sum(dE)

                if maximum_energy > 0:
                    data_fenergy.append(temp_genergy[0])
                    data_senergy.append(temp_genergy[1])
                    data_aenergy.append(temp_genergy[2])
                    data_pfenergy.append(temp_genergy[0] / temp_genergy[2])
                    data_psenergy.append(temp_genergy[1] / temp_genergy[2])
                    data_lctes.append(maximum_energy)
                    data_lctls.append(maximum_energy_length)
                    data_closestedge.append(closest_edge)
                    data_theta.append(theta)
                else:
                    data_fenergy.append(0)
                    data_senergy.append(0)
                    data_aenergy.append(0)
                    data_pfenergy.append(0)
                    data_psenergy.append(0)
                    data_lctes.append(0)
                    data_lctls.append(0)
                    data_closestedge.append(0)
                    data_theta.append(0)
                
        f.close()
        g.close()

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/BRN_segment_data.npz', fenergy=data_fenergy, senergy=data_senergy, aenergy=data_aenergy,
                        pfenergy=data_pfenergy, psenergy=data_psenergy, lctes=data_lctes, lctls=data_lctls, closestedge=data_closestedge, theta=data_theta)
    
    print("Data successfully written to file BRN_segment_data.npz!")