import h5py
import json
import numpy as np

infile_dir = '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/NueArCC'
outfile_dir = 'graph_data'
data = {"electron": {"mult": [], "energy": [], "diry": [], "dirz": []}, "photon": {"mult": [], "energy": [], "diry": [], "dirz": []}, "proton": {"energy": [], "diry": [], "dirz": []}, "neutron": {"energy": [], "diry": [], "dirz": []}, "other": {}}

if __name__ == "__main__":
    for i in range(10):
        print("Loading file " + str(i + 1) + "/10...")
        f = h5py.File(infile_dir + '/nueArCC_sns_g4_' + format(i, "04") + '.h5', 'r')

        event_id = f['trajectories'][0]['event_id']
        n_pelec = 0
        n_ppho = 0

        for traj in f['trajectories']:  
            if traj['event_id'] > event_id:
                data['electron']['mult'].append(n_pelec)
                data['photon']['mult'].append(n_ppho)
                event_id = traj['event_id']
                n_pelec = 0
                n_ppho = 0

            if traj['parent_id'] == -1:
                # Primary electron data collection
                if traj['pdg_id'] == 11:
                    n_pelec += 1
                    data['electron']['energy'].append(traj['E_start'])
                    data['electron']['diry'].append(traj['pxyz_start'][1] / np.sqrt(np.sum(np.square(traj['pxyz_start']))))
                    try:
                        if (traj['pxyz_start'][0] >= 0):
                            data['electron']['dirz'].append(np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2 + traj['pxyz_start'][2]**2)))
                        else:
                            data['electron']['dirz'].append(2*np.pi - np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2
                                                                                                                + traj['pxyz_start'][2]**2)))
                    except ZeroDivisionError:
                        data['electron']['dirz'].append(0)

                # Primary photon data collection
                elif traj['pdg_id'] == 22:
                    n_ppho += 1
                    data['photon']['energy'].append(traj['E_start'])
                    data['photon']['diry'].append(traj['pxyz_start'][1] / np.sqrt(np.sum(np.square(traj['pxyz_start']))))
                    try:
                        if (traj['pxyz_start'][0] >= 0):
                            data['photon']['dirz'].append(np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2 + traj['pxyz_start'][2]**2)))
                        else:
                            data['photon']['dirz'].append(2*np.pi - np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2
                                                                                                              + traj['pxyz_start'][2]**2)))
                    except ZeroDivisionError:
                        data['photon']['dirz'].append(0)

                # Primary proton data collection
                elif traj['pdg_id'] == 2212:
                    data['proton']['energy'].append(traj['E_start'] - 938.27208943)
                    data['proton']['diry'].append(traj['pxyz_start'][1] / np.sqrt(np.sum(np.square(traj['pxyz_start']))))
                    try:
                        if (traj['pxyz_start'][0] >= 0):
                            data['proton']['dirz'].append(np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2 + traj['pxyz_start'][2]**2)))
                        else:
                            data['proton']['dirz'].append(2*np.pi - np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2
                                                                                                              + traj['pxyz_start'][2]**2)))
                    except ZeroDivisionError:
                        data['proton']['dirz'].append(0)

                # Primary neutron data collection
                elif traj['pdg_id'] == 2112:
                    data['neutron']['energy'].append(traj['E_start'] - 939.56542194)
                    data['neutron']['diry'].append(traj['pxyz_start'][1] / np.sqrt(np.sum(np.square(traj['pxyz_start']))))
                    try:
                        if (traj['pxyz_start'][0] >= 0):
                            data['neutron']['dirz'].append(np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2 + traj['pxyz_start'][2]**2)))
                        else:
                            data['neutron']['dirz'].append(2*np.pi - np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2
                                                                                                               + traj['pxyz_start'][2]**2)))
                    except ZeroDivisionError:
                        data['neutron']['dirz'].append(0)

                # Other particle--save for future reference
                else:
                    if traj['pdg_id'] in data['other']:
                        data['other'][traj['pdg_id']] += 1
                    else:
                        data['other'][traj['pdg_id']] = 1

        data['electron']['mult'].append(n_pelec)
        data['photon']['mult'].append(n_ppho)
        f.close()

    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/signal_primary_particles.npz', emult=data['electron']['mult'], eenergy=data['electron']['energy'], 
                        ediry=data['electron']['diry'], edirz=data['electron']['dirz'], phmult=data['photon']['mult'], phenergy=data['photon']['energy'],
                        phdiry=data['photon']['diry'], phdirz=data['photon']['dirz'], prenergy=data['proton']['energy'], prdiry=data['proton']['diry'], 
                        prdirz=data['proton']['dirz'], nenergy=data['neutron']['energy'], ndiry=data['neutron']['diry'], ndirz=data['neutron']['dirz'],
                        other_keys=np.array(list(data['other'].keys())), other_vals=np.array(list(data['other'].values())))
    
    print("Data successfully written to file signal_primary_particles.npz!")

