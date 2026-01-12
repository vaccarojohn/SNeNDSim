import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume

infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/Cosmics_Overlay_processed'
outfile_dir = 'graph_data'
    
if __name__ == "__main__":
    data_prenergies = []
    data_prlengths = []
    data_lctes = []
    data_lctls = []
    edata_particles = {}
    data_lengths = []
    ldata_particles = {}
    data_fenergy = []
    data_senergy = []
    data_aenergy = []
    data_pfenergy = []
    data_psenergy = []

    # HOG-related quantities
    data_numhog = []
    data_hogdep = []

    for i in range(1200):
        print("Loading file " + str(i + 1) + "/1200...")
        f = h5py.File(infile_dir + '/CosmicFlux_g4_' + format(i, "04") + '-processed.h5', 'r')

        event_id = f['segments'][0]['event_id']
        temp_particles = {}
        temp_energies = {}
        temp_lengths = {}
        temp_totalenergy = [0, 0, 0]
        temp_inDet = False
        temp_hogtpc = []
        temp_hogdep = 0

        for seg in f['segments']:
            if seg['event_id'] > event_id:
                if temp_inDet:
                    greatest_energy_dep = 0
                    longest_path = 0

                    # Search for the longest path and highest energy deposit for each event
                    for traj_id in temp_particles:
                        if greatest_energy_dep not in temp_energies or temp_energies[traj_id] > temp_energies[greatest_energy_dep]:
                            greatest_energy_dep = traj_id

                        if longest_path not in temp_lengths or temp_lengths[traj_id] > temp_lengths[longest_path]:
                            longest_path = traj_id

                    # Save the longest path and highest energy deposit (in active volume), as well as the particles responsible
                    if temp_energies[greatest_energy_dep] > 0:
                        data_lctes.append(temp_energies[greatest_energy_dep])
                        data_lctls.append(temp_lengths[greatest_energy_dep])
                        if (edata_particles.get(temp_particles[greatest_energy_dep])):
                            edata_particles[temp_particles[greatest_energy_dep]] += 1
                        else:
                            edata_particles[temp_particles[greatest_energy_dep]] = 1
                    else:
                        data_lctes.append(0)
                        data_lctls.append(0)

                    if temp_lengths[longest_path] > 0:
                        data_lengths.append(temp_lengths[longest_path])
                        if (ldata_particles.get(temp_particles[longest_path])):
                            ldata_particles[temp_particles[longest_path]] += 1
                        else:
                            ldata_particles[temp_particles[longest_path]] = 1
                    else:
                        data_lengths.append(0)
                    
                    if temp_totalenergy[2] > 0:
                        # Save primary electron trajectory energy and length
                        if 0 in temp_particles:
                            data_prenergies.append(temp_energies[0])
                            data_prlengths.append(temp_lengths[0])
                        else:
                            data_prenergies.append(0)
                            data_prlengths.append(0)
                    
                        data_fenergy.append(temp_totalenergy[0])
                        data_senergy.append(temp_totalenergy[1])
                        data_aenergy.append(temp_totalenergy[2])
                        data_pfenergy.append(temp_totalenergy[0] / temp_totalenergy[2])
                        data_psenergy.append(temp_totalenergy[1] / temp_totalenergy[2])
                        data_numhog.append(len(temp_hogtpc))
                        data_hogdep.append(temp_hogdep)
                    else:
                        data_prenergies.append(0)
                        data_prlengths.append(0)
                        data_fenergy.append(0)
                        data_senergy.append(0)
                        data_aenergy.append(0)
                        data_pfenergy.append(0)
                        data_psenergy.append(0)
                        data_numhog.append(0)
                        data_hogdep.append(0)
                
                temp_particles = {}
                temp_energies = {}
                temp_lengths = {}
                temp_totalenergy = [0, 0, 0]
                temp_inDet = False
                temp_hogtpc = []
                temp_hogdep = 0
                
                event_id = seg['event_id']

            # Save the distance traveled inside the active, signal, and fiducial volumes
            in_active = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                    seg['y_start'], seg['y_end'], 
                                                    seg['z_start'], seg['z_end'])
            in_signal = get_length_in_signal_volume(seg['x_start'], seg['x_end'], 
                                                    seg['y_start'], seg['y_end'], 
                                                    seg['z_start'], seg['z_end'])
            in_fiducial = get_length_in_fiducial_volume(seg['x_start'], seg['x_end'], 
                                                        seg['y_start'], seg['y_end'], 
                                                        seg['z_start'], seg['z_end'])

            if in_active != 0:
                temp_inDet = True
                
            # Add the energy deposited in the active, signal, and fiducial volumes for this segment
            temp_totalenergy[0] += seg['dEdx'] * in_fiducial

            temp_totalenergy[1] += seg['dEdx'] * in_signal

            temp_totalenergy[2] += seg['dEdx'] * in_active

            # Add the energy deposited and segment length (in active volume) to determine the trajectory with the highest energy deposition and longest path
            if temp_particles.get(seg['traj_id']):
                temp_energies[seg['traj_id']] += seg['dEdx'] * in_active
                temp_lengths[seg['traj_id']] += in_active
            else:
                temp_particles[seg['traj_id']] = seg['pdg_id']
                temp_energies[seg['traj_id']] = seg['dEdx'] * in_active
                temp_lengths[seg['traj_id']] = in_active

            # Add HOG-related quantities
            if in_active != 0 and seg['vertex_id'] != 0:
                temp_hogdep += seg['dEdx'] * in_active
                if seg['vertex_id'] not in temp_hogtpc:
                    temp_hogtpc.append(seg['vertex_id'])

        # Process final event in file
        if temp_inDet:
            greatest_energy_dep = 0
            longest_path = 0

            # Search for the longest path and highest energy deposit for each event
            for traj_id in temp_particles:
                if greatest_energy_dep not in temp_energies or temp_energies[traj_id] > temp_energies[greatest_energy_dep]:
                    greatest_energy_dep = traj_id

                if longest_path not in temp_lengths or temp_lengths[traj_id] > temp_lengths[longest_path]:
                    longest_path = traj_id

            # Save the longest path and highest energy deposit (in active volume), as well as the particles responsible
            if temp_energies[greatest_energy_dep] > 0:
                data_lctes.append(temp_energies[greatest_energy_dep])
                data_lctls.append(temp_lengths[greatest_energy_dep])
                if (edata_particles.get(temp_particles[greatest_energy_dep])):
                    edata_particles[temp_particles[greatest_energy_dep]] += 1
                else:
                    edata_particles[temp_particles[greatest_energy_dep]] = 1
            else:
                data_lctes.append(0)
                data_lctls.append(0)

            if temp_lengths[longest_path] > 0:
                data_lengths.append(temp_lengths[longest_path])
                if (ldata_particles.get(temp_particles[longest_path])):
                    ldata_particles[temp_particles[longest_path]] += 1
                else:
                    ldata_particles[temp_particles[longest_path]] = 1
            else:
                data_lengths.append(0)
                    
            if temp_totalenergy[2] > 0:
                # Save primary electron trajectory energy and length
                if 0 in temp_particles:
                    data_prenergies.append(temp_energies[0])
                    data_prlengths.append(temp_lengths[0])
                else:
                    data_prenergies.append(0)
                    data_prlengths.append(0)
                    
                data_fenergy.append(temp_totalenergy[0])
                data_senergy.append(temp_totalenergy[1])
                data_aenergy.append(temp_totalenergy[2])
                data_pfenergy.append(temp_totalenergy[0] / temp_totalenergy[2])
                data_psenergy.append(temp_totalenergy[1] / temp_totalenergy[2])
                data_numhog.append(len(temp_hogtpc))
                data_hogdep.append(temp_hogdep)
            else:
                data_prenergies.append(0)
                data_prlengths.append(0)
                data_fenergy.append(0)
                data_senergy.append(0)
                data_aenergy.append(0)
                data_pfenergy.append(0)
                data_psenergy.append(0)
                data_numhog.append(0)
                data_hogdep.append(0)

        f.close()

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/cosmics_segment_data.npz', fenergy=data_fenergy, senergy=data_senergy, aenergy=data_aenergy, 
                        pfenergy=data_pfenergy, psenergy=data_psenergy, prenergies=data_prenergies, prlengths=data_prlengths, lctes=data_lctes, 
                        lctls=data_lctls, lengths=data_lengths, eparticles=np.array(list(edata_particles.keys())), 
                        ecounts=np.array(list(edata_particles.values())), lparticles=np.array(list(ldata_particles.keys())), 
                        lcounts=np.array(list(ldata_particles.values())), numhog=data_numhog, hogdep=data_hogdep)
    
    print("Data successfully written to file cosmics_segment_data.npz!")