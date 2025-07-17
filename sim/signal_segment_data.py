import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume

infile_dir = '/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/NueArCC'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_energies = []
    edata_particles = {}
    data_lengths = []
    ldata_particles = {}
    data_fenergy = []
    data_senergy = []

    for i in range(10):
        print("Loading file " + str(i + 1) + "/10...")
        f = h5py.File(infile_dir + '/nueArCC_sns_yDir_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_particles = {}
        temp_energies = {}
        temp_lengths = {}
        temp_totalenergy = [0, 0, 0]

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                greatest_energy_dep = 0
                longest_path = 0

                # Search for the longest path and highest energy deposit for each event
                for traj_id in temp_particles:
                    if temp_energies[traj_id] > temp_energies[greatest_energy_dep]:
                        greatest_energy_dep = traj_id

                    if temp_lengths[traj_id] > temp_lengths[longest_path]:
                        longest_path = traj_id

                # Save the particles responsible for the longest path and highest energy deposit
                if (edata_particles.get(temp_particles[greatest_energy_dep])):
                    edata_particles[temp_particles[greatest_energy_dep]] += 1
                else:
                    edata_particles[temp_particles[greatest_energy_dep]] = 1

                if (ldata_particles.get(temp_particles[longest_path])):
                    ldata_particles[temp_particles[longest_path]] += 1
                else:
                    ldata_particles[temp_particles[longest_path]] = 1

                # Save the greatest energy deposit, longest path length, and percentage of energy deposited in the fiducial/signal volumes
                data_energies.append(temp_energies[greatest_energy_dep])
                data_lengths.append(temp_lengths[longest_path])

                if temp_totalenergy[2] == 0:
                    data_fenergy.append(0)
                    data_senergy.append(0)
                else:
                    data_fenergy.append(temp_totalenergy[0] / temp_totalenergy[2])
                    data_senergy.append(temp_totalenergy[1] / temp_totalenergy[2])

                
                temp_particles = {}
                temp_energies = {}
                temp_lengths = {}
                temp_totalenergy = [0, 0, 0]
                
                event_id = seg['event_id']

            # Add the energy deposited in the active, signal, and fiducial volumes for this segment
            temp_totalenergy[0] += seg['dEdx'] * get_length_in_fiducial_volume(seg['x_start'], seg['x_end'], 
                                                                               seg['y_start'], seg['y_end'], 
                                                                               seg['z_start'], seg['z_end'])

            temp_totalenergy[1] += seg['dEdx'] * get_length_in_signal_volume(seg['x_start'], seg['x_end'], 
                                                                             seg['y_start'], seg['y_end'], 
                                                                             seg['z_start'], seg['z_end'])

            temp_totalenergy[2] += seg['dEdx'] * get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                                             seg['y_start'], seg['y_end'], 
                                                                             seg['z_start'], seg['z_end'])

            # Add the energy deposited and segment length to determine the trajectory with the highest energy deposition and longest path later
            if temp_particles.get(seg['traj_id']):
                temp_energies[seg['traj_id']] += seg['dE']
                temp_lengths[seg['traj_id']] += seg['dx']
            else:
                temp_particles[seg['traj_id']] = seg['pdg_id']
                temp_energies[seg['traj_id']] = seg['dE']
                temp_lengths[seg['traj_id']] = seg['dx']

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/signal_segment_data.npz', fenergy=data_fenergy, senergy=data_senergy, energies=data_energies,
                        lengths=data_lengths, eparticles=np.array(list(edata_particles.keys())), ecounts=np.array(list(edata_particles.values())), 
                        lparticles=np.array(list(ldata_particles.keys())), lcounts=np.array(list(ldata_particles.values())))
    
    print("Data successfully written to file signal_segment_data.npz!")