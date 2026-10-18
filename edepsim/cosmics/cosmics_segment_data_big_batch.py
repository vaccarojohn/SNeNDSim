import sys
import h5py
import numpy as np
sys.path.append('/sdf/home/j/jvaccaro/SNeNDSim')
from helper_functions import edge_dist, get_theta, get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume

infile_dir = '/sdf/data/neutrino/eob11235/SNeND/SNeNDSens/20x20BatchOutput'
#infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/Cosmics_processed'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_lctes = []
    data_lctls = []
    data_closestedge = []
    data_senergy = []
    data_aenergy = []
    data_psenergy = []
    
    for i in range(int(sys.argv[1]), int(sys.argv[2])):
        print("Loading file " + str(i + 1) + "/12000...")
        f = h5py.File(infile_dir + '/CosmicFlux_g4_' + format(i, "05") + '.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_particles = {}
        temp_energies = {}
        temp_lengths = {}
        temp_closestedge = {}
        temp_totalenergy = [0, 0]
        temp_inDet = False

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                if temp_inDet:
                    greatest_energy_dep = 0

                    # Search for the longest path and highest energy deposit for each event
                    for traj_id in temp_particles:
                        if greatest_energy_dep not in temp_energies or temp_energies[traj_id] > temp_energies[greatest_energy_dep]:
                            greatest_energy_dep = traj_id

                    # Save the longest path and highest energy deposit (in active volume), as well as the particles responsible
                    if temp_energies.get(greatest_energy_dep) and temp_energies[greatest_energy_dep] > 0:
                        data_lctes.append(temp_energies[greatest_energy_dep])
                        data_lctls.append(temp_lengths[greatest_energy_dep])
                        data_closestedge.append(temp_closestedge[greatest_energy_dep])
                    else:
                        data_lctes.append(0)
                        data_lctls.append(0)
                        data_closestedge.append(0)
                    
                    if temp_totalenergy[1] > 0:
                        data_senergy.append(temp_totalenergy[0])
                        data_aenergy.append(temp_totalenergy[1])
                        data_psenergy.append(temp_totalenergy[0] / temp_totalenergy[1])
                    else:
                        data_senergy.append(0)
                        data_aenergy.append(0)
                        data_psenergy.append(0)
                
                temp_energies = {}
                temp_lengths = {}
                temp_closestedge = {}
                temp_totalenergy = [0, 0]
                temp_inDet = False
                
                event_id = seg['event_id']

            # Save the distance traveled inside the active, signal, and fiducial volumes
            in_active = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                    seg['y_start'], seg['y_end'], 
                                                    seg['z_start'], seg['z_end'])
            in_signal = get_length_in_signal_volume(seg['x_start'], seg['x_end'], 
                                                    seg['y_start'], seg['y_end'], 
                                                    seg['z_start'], seg['z_end'])

            if in_active != 0:
                temp_inDet = True
                
            # Add the energy deposited in the active, signal, and fiducial volumes for this segment
            temp_totalenergy[0] += seg['dEdx'] * in_signal

            temp_totalenergy[1] += seg['dEdx'] * in_active

            # Add the energy deposited and segment length (in active volume) to determine the trajectory with the highest energy deposition and longest path
            if temp_particles.get(seg['traj_id']):
                temp_energies[seg['traj_id']] += seg['dEdx'] * in_active
                temp_lengths[seg['traj_id']] += in_active
                temp_closestedge[seg['traj_id']] = min(temp_closestedge[seg['traj_id']], 
                                                       edge_dist(seg['x_start'], seg['x_end'],                
                                                                 seg['y_start'], seg['y_end'],
                                                                 seg['z_start'], seg['z_end'],
                                                                 30, 25, 30))
            else:
                temp_energies[seg['traj_id']] = seg['dEdx'] * in_active
                temp_lengths[seg['traj_id']] = in_active

                temp_closestedge[seg['traj_id']] = edge_dist(seg['x_start'], seg['x_end'],
                                                                seg['y_start'], seg['y_end'],
                                                                seg['z_start'], seg['z_end'],
                                                                30, 25, 30)

        # Process final event in file
        if temp_inDet:
            greatest_energy_dep = 0

            # Search for the longest path and highest energy deposit for each event
            for traj_id in temp_particles:
                if greatest_energy_dep not in temp_energies or temp_energies[traj_id] > temp_energies[greatest_energy_dep]:
                    greatest_energy_dep = traj_id

            # Save the longest path and highest energy deposit (in active volume), as well as the particles responsible
            if temp_energies.get(greatest_energy_dep) and temp_energies[greatest_energy_dep] > 0:
                data_lctes.append(temp_energies[greatest_energy_dep])
                data_lctls.append(temp_lengths[greatest_energy_dep])
                data_closestedge.append(temp_closestedge[greatest_energy_dep])
            else:
                data_lctes.append(0)
                data_lctls.append(0)
                data_closestedge.append(0)
                    
            if temp_totalenergy[1] > 0:
                data_senergy.append(temp_totalenergy[0])
                data_aenergy.append(temp_totalenergy[1])
                data_psenergy.append(temp_totalenergy[0] / temp_totalenergy[1])
            else:
                data_senergy.append(0)
                data_aenergy.append(0)
                data_psenergy.append(0)

        f.close()

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/cosmics_segment_data_big_batch_' + str(i+1) + '.npz',
                                senergy=data_senergy, aenergy=data_aenergy,
                                psenergy=data_psenergy, lctes=data_lctes, lctls=data_lctls, 
                                closestedge=data_closestedge)
    
    print("Data successfully written to file cosmics_segment_data.npz!")