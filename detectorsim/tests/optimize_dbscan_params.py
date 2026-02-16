import h5py
import numpy as np
from gampixpy import config
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from helper_functions import get_length_in_fiducial_volume, get_length_in_signal_volume, get_length_in_active_volume
from test_segment_info import Trace, inverseBirksModel

metadata_infile_dir = '/sdf/home/j/jvaccaro/edepsim/signal/graph_data'
edepsim_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/NueArCC'
gampix_infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/gampixpy/NueArCC'
outfile_dir = 'graph_data'
physics_config = config.default_physics_params

DB_EPSILON = [0.7]
DB_MIN_SAMPLES = [3]
FILE = 0

data = np.zeros((1, 1, 27, 10000), dtype=np.float64)

def main():
    f = h5py.File(edepsim_infile_dir + '/nueArCC_sns_g4_' + format(FILE, "04") + '.h5', 'r')
    g = h5py.File(gampix_infile_dir + '/nueArCC_sns_g4_gampixpy_' + format(FILE, "04") + '.h5', 'r')

    for epsilon_id, epsilon in enumerate(DB_EPSILON):
        print("EPSILON=" + str(epsilon))
        for min_samples_id, min_samples in enumerate(DB_MIN_SAMPLES):
            print("MIN_SAMPLES=" + str(min_samples))
            for event in range(10000):
                if event % 500 == 0:
                    print("EVENT=" + str(event))
                gampixpy_traces = []
                edepsim_traces = {}
                dbscan_traces = []
            
                # Find charge readout data from GAMPixPy and recalculate `hit x` based on vertex timing
                vertex_mask = f['vertices']['event_id'] == event
                segment_mask = f['segments']['event_id'] == event
                event_pixel_hit_mask = g['pixel_hits']['event id'] == event
                event_pixel_hits = g['pixel_hits'][event_pixel_hit_mask]
            
                drift_time = event_pixel_hits['hit t'] - f['vertices'][vertex_mask][0]['t_vert']
                tpc_index_selector = 2 * event_pixel_hits['tpc index'].astype(int) - 1 # -1 for TPC index 0, 1 for TPC index 1
                hit_x = (30 - drift_time * physics_config['charge_drift']['drift_speed']) * tpc_index_selector
            
                # Calculate charge deposited after recombination by accounting for electron lifetime in LAr
                charge = event_pixel_hits['hit charge'] * np.exp(drift_time / physics_config['charge_drift']['electron_lifetime'])
            
                #print("----------------------------------------------------------------------------------------------------------------------")
                
                # Find out EdepSim hitSegments contributing to EdepSim trajectories
                for i, seg in enumerate(f['segments'][segment_mask]):
                    in_active = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                            seg['y_start'], seg['y_end'], 
                                                            seg['z_start'], seg['z_end'])
                    
                    if edepsim_traces.get(seg['traj_id']):
                        edepsim_traces[seg['traj_id']].increment_mask(i)
                        edepsim_traces[seg['traj_id']].add(np.array([seg['segment_id']]), np.array([seg['dEdx'] * in_active]), 1, (seg['x_start'] + seg['x_end'])/2,
                                                                    (seg['y_start'] + seg['y_end'])/2, (seg['z_start'] + seg['z_end'])/2, (seg['t_start'] + seg['t_end'])/2)
                    else:
                        trace = Trace()
                        trace.increment_mask(i)
                        trace.add(np.array([seg['segment_id']]), np.array([seg['dEdx'] * in_active]), 1, (seg['x_start'] + seg['x_end'])/2,
                                           (seg['y_start'] + seg['y_end'])/2, (seg['z_start'] + seg['z_end'])/2, (seg['t_start'] + seg['t_end'])/2)
                        edepsim_traces[seg['traj_id']] = trace
            
                #print("EdepSim trajectory truth data: " + str([str(traj_id) + ": " + str(edepsim_traces[traj_id].segs) for traj_id in edepsim_traces]))
            
                # Find total energy deposited in TPC and leading charged trajectory
                maximum_energy = -1
                maximum_energy_id = -1
                aenergy = 0

                # Data for comparison with DBSCAN
                aenergy_edepsim = 0
                lcte_edepsim = 0
                center_edepsim = [0, 0, 0, 0]
                lct_segments_edepsim = []
                
                for traj_id in edepsim_traces:
                    edepsim_traces[traj_id].fill_mask(len(f['segments'][segment_mask]))
                    dE = sum(edepsim_traces[traj_id].contrib)
                    aenergy += dE
                    
                    if dE > maximum_energy:
                        maximum_energy = dE
                        maximum_energy_id = traj_id

                # Save data for comparison with DBSCAN
                if maximum_energy != -1:
                    aenergy_edepsim = aenergy
                    lcte_edepsim = maximum_energy
                    center_edepsim = edepsim_traces[maximum_energy_id].calculate_center()
                    lct_segments_edepsim = edepsim_traces[maximum_energy_id].segs
            
                """print("The total energy deposited in the TPC was: " + str(aenergy) + " MeV.")
                print("EdepSim identified the leading charged trajectory as: #" + str(maximum_energy_id))
                print("The center of this trajectory was: " + str(edepsim_traces[maximum_energy_id].calculate_center()) + ".")
                print("The total energy deposition from this trajectory was: " + str(maximum_energy) + " MeV.")
            
                print("----------------------------------------------------------------------------------------------------------------------")"""
            
                # Find out EdepSim hitSegments contributing to GAMPixPy traces
                for i, hit in enumerate(event_pixel_hits):
                    trace_id = -1
                    to_remove = []
                    for j, trace in enumerate(gampixpy_traces):
                        if trace.correct_trace(hit['label']):
                            if trace_id == -1:
                                trace.add(hit['label'], hit['attribution'], charge[i], hit_x[i], hit['hit y'], hit['hit z'], hit['hit t'])
                                trace.increment_mask(i)
                                trace_id = j
                            elif j not in to_remove:
                                gampixpy_traces[trace_id].merge(trace)
                                to_remove.append(j)
                                
                    for j in sorted(to_remove, reverse=True):
                        gampixpy_traces.pop(j)
                            
                    if trace_id == -1:
                        trace = Trace()
                        trace.increment_mask(i)
                        trace.add(hit['label'], hit['attribution'], charge[i], hit_x[i], hit['hit y'], hit['hit z'], hit['hit t'])
                        gampixpy_traces.append(trace)
            
                #print("GAMPixPy trace segments data: " + str([trace.segs for trace in gampixpy_traces]))
            
                # Find total energy deposited in TPC and leading charged trace
                maximum_energy = -1
                maximum_energy_id = -1
                aenergy = 0
        
                # Data for comparison with DBSCAN
                aenergy_gampixpy = 0
                lcte_gampixpy = 0
                center_gampixpy = [0, 0, 0, 0]
                lct_segments_gampixpy = []
                
                for i, trace in enumerate(gampixpy_traces):
                    trace.fill_mask(len(event_pixel_hits))
                    trace_mask = trace.mask_numpy()
                    dQ = sum(trace.contrib) + trace.noise
            
                    pca = PCA(n_components=1)
                    transformed_data = pca.fit_transform(np.column_stack((hit_x[trace_mask], event_pixel_hits[trace_mask]['hit y'], 
                                                                          event_pixel_hits[trace_mask]['hit z'])))
            
                    start = np.argmin(transformed_data)
                    end = np.argmax(transformed_data)
                    dx = np.sqrt((hit_x[trace_mask][end] - hit_x[trace_mask][start])**2
                                + (event_pixel_hits[trace_mask][end]['hit y'] - event_pixel_hits[trace_mask][start]['hit y'])**2
                                + (event_pixel_hits[trace_mask][end]['hit z'] - event_pixel_hits[trace_mask][start]['hit z'])**2)

                    dE = 0

                    if dx > 0:
                        dEdx = inverseBirksModel(dQ / dx)
                        dE = dEdx * dx
                        aenergy += dE
                    
                    if dE > maximum_energy:
                        maximum_energy = dE
                        maximum_energy_id = i
        
                # Save data for comparison with DBSCAN
                if maximum_energy != -1:
                    aenergy_gampixpy = aenergy
                    lcte_gampixpy = maximum_energy
                    center_gampixpy = gampixpy_traces[maximum_energy_id].calculate_center()
                    lct_segments_gampixpy = gampixpy_traces[maximum_energy_id].segs
            
                """print("The total energy deposited in the TPC was: " + str(aenergy) + " MeV.")
                print("GAMPixPy identified the leading charged trace as: #" + str(maximum_energy_id))
                print("The center of this trace was: " + str(gampixpy_traces[maximum_energy_id].calculate_center()) + ".")
                print("The total energy deposition from this trace was: " + str(maximum_energy) + " MeV.")
            
                print("----------------------------------------------------------------------------------------------------------------------")"""
            
                # Find out EdepSim hitSegments contributing to DBSCAN-identified traces
                db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(np.column_stack((hit_x, event_pixel_hits['hit y'], event_pixel_hits['hit z'], event_pixel_hits['hit t'])))
                dbscan_trace_labels = set(db.labels_)
                for k in dbscan_trace_labels:
                    if k == -1:
                        continue
            
                    trace_member_mask = db.labels_ == k
                    trace = Trace()
                    trace.member_mask = list(trace_member_mask.astype(int))
            
                    for i, hit in enumerate(event_pixel_hits[trace_member_mask]):
                        trace.add(hit['label'], hit['attribution'], charge[i], hit_x[i], hit['hit y'], hit['hit z'], hit['hit t'])
            
                    dbscan_traces.append(trace)
            
                #print("DBSCAN trace segments data: " + str([trace.segs for trace in dbscan_traces]))
            
                # Find total energy deposited in TPC and leading charged trace
                maximum_energy = -1
                maximum_energy_id = -1
                aenergy = 0
        
                # Data for comparison with GAMPixPy
                aenergy_dbscan = 0
                lcte_dbscan = 0
                center_dbscan = [0, 0, 0, 0]
                lct_segments_dbscan = []
                
                for i, trace in enumerate(dbscan_traces):
                    trace.fill_mask(len(event_pixel_hits))
                    trace_mask = trace.mask_numpy()
                    dQ = sum(trace.contrib) + trace.noise
            
                    pca = PCA(n_components=1)
                    transformed_data = pca.fit_transform(np.column_stack((hit_x[trace_mask], event_pixel_hits[trace_mask]['hit y'], 
                                                                          event_pixel_hits[trace_mask]['hit z'])))
            
                    start = np.argmin(transformed_data)
                    end = np.argmax(transformed_data)
                    dx = np.sqrt((hit_x[trace_mask][end] - hit_x[trace_mask][start])**2
                                + (event_pixel_hits[trace_mask][end]['hit y'] - event_pixel_hits[trace_mask][start]['hit y'])**2
                                + (event_pixel_hits[trace_mask][end]['hit z'] - event_pixel_hits[trace_mask][start]['hit z'])**2)

                    dE = 0
                    if dx > 0:  
                        dEdx = inverseBirksModel(dQ / dx)
                        dE = dEdx * dx
                        aenergy += dE
                    
                    if dE > maximum_energy:
                        maximum_energy = dE
                        maximum_energy_id = i
        
                # Save data for comparison with GAMPixPy
                if maximum_energy != -1:
                    aenergy_dbscan = aenergy
                    lcte_dbscan = maximum_energy
                    center_dbscan = dbscan_traces[maximum_energy_id].calculate_center()
                    lct_segments_dbscan = dbscan_traces[maximum_energy_id].segs
        
                # Save to data
                data[epsilon_id, min_samples_id, 0, event] = aenergy_dbscan
                data[epsilon_id, min_samples_id, 1, event] = aenergy_gampixpy
                data[epsilon_id, min_samples_id, 2, event] = aenergy_edepsim

                data[epsilon_id, min_samples_id, 3, event] = lcte_dbscan
                data[epsilon_id, min_samples_id, 4, event] = lcte_gampixpy
                data[epsilon_id, min_samples_id, 5, event] = lcte_edepsim
                    
                data[epsilon_id, min_samples_id, 6, event] = center_dbscan[0]
                data[epsilon_id, min_samples_id, 7, event] = center_gampixpy[0]
                data[epsilon_id, min_samples_id, 8, event] = center_edepsim[0]

                data[epsilon_id, min_samples_id, 9, event] = center_dbscan[1]
                data[epsilon_id, min_samples_id, 10, event] = center_gampixpy[1]
                data[epsilon_id, min_samples_id, 11, event] = center_edepsim[1]
                
                data[epsilon_id, min_samples_id, 12, event] = center_dbscan[2]
                data[epsilon_id, min_samples_id, 13, event] = center_gampixpy[2]
                data[epsilon_id, min_samples_id, 14, event] = center_edepsim[2]
                
                data[epsilon_id, min_samples_id, 15, event] = center_dbscan[3]
                data[epsilon_id, min_samples_id, 16, event] = center_gampixpy[3]
                data[epsilon_id, min_samples_id, 17, event] = center_edepsim[3]

                data[epsilon_id, min_samples_id, 18, event] = len(set(lct_segments_gampixpy) - set(lct_segments_dbscan))
                data[epsilon_id, min_samples_id, 19, event] = len(set(lct_segments_dbscan) - set(lct_segments_gampixpy))

                data[epsilon_id, min_samples_id, 20, event] = len(set(lct_segments_edepsim) - set(lct_segments_dbscan))
                data[epsilon_id, min_samples_id, 21, event] = len(set(lct_segments_dbscan) - set(lct_segments_edepsim))

                data[epsilon_id, min_samples_id, 22, event] = len(set(lct_segments_gampixpy) - set(lct_segments_edepsim))
                data[epsilon_id, min_samples_id, 23, event] = len(set(lct_segments_edepsim) - set(lct_segments_gampixpy))

                data[epsilon_id, min_samples_id, 24, event] = len(lct_segments_dbscan)
                data[epsilon_id, min_samples_id, 25, event] = len(lct_segments_gampixpy)
                data[epsilon_id, min_samples_id, 26, event] = len(lct_segments_edepsim)
            
                """print("The total energy deposited in the TPC was: " + str(aenergy) + " MeV.")
                print("DBSCAN identified the leading charged trace as: #" + str(maximum_energy_id))
                print("The center of this trace was: " + str(dbscan_traces[maximum_energy_id].calculate_center()) + ".")
                print("The total energy deposition from this trajectory was: " + str(maximum_energy) + " MeV.")
            
                print("----------------------------------------------------------------------------------------------------------------------")"""
    
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/dbscan_final_data.npz', data=data)

if __name__ == "__main__":
    main()