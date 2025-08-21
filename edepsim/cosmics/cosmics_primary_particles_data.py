import h5py
import numpy as np

infile_dir = '/sdf/data/neutrino/jvaccaro/edepsim/Cosmics_unprocessed'
outfile_dir = 'graph_data'
data = {"muon": {"energy": [], "diry": [], "dirz": []}, "electron": {"mult": [], "energy": [], "diry": [], "dirz": []}, "photon": {"mult": [], "energy": [], "diry": [], "dirz": []}, "other": {}}

if __name__ == "__main__":
    for i in range(1200):
        print("Loading file " + str(i + 1) + "/1200...")
        f = h5py.File(infile_dir + '/CosmicFlux_g4_' + format(i, "04") + '.h5', 'r')

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

            # Primary muon data collection
            if traj['parent_id'] == -1:
                data['muon']['energy'].append(traj['E_start'] - 105.6583755)
                data['muon']['diry'].append(traj['pxyz_start'][1] / np.sqrt(np.sum(np.square(traj['pxyz_start']))))
                try:
                    if (traj['pxyz_start'][0] >= 0):
                        data['muon']['dirz'].append(np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2 + traj['pxyz_start'][2]**2)))
                    else:
                        data['muon']['dirz'].append(2*np.pi - np.arccos(traj['pxyz_start'][2] / np.sqrt(traj['pxyz_start'][0]**2
                                                                                                + traj['pxyz_start'][2]**2)))
                except ZeroDivisionError:
                    data['muon']['dirz'].append(0)
                    
            elif traj['parent_id'] == 0:
                # Muon-induced electron data collection
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

                # Muon-induced photon data collection
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

                # Other particle--save for future reference
                else:
                    if traj['pdg_id'] in data['other']:
                        data['other'][traj['pdg_id']] += 1
                    else:
                        data['other'][traj['pdg_id']] = 1

        f.close()
                        
    print("Writing to output...")
    np.savez_compressed(outfile_dir + '/cosmics_primary_particles.npz', menergy=data['muon']['energy'], mdiry=data['muon']['diry'], mdirz=data['muon']['dirz'], 
                        emult=data['electron']['mult'], eenergy=data['electron']['energy'], ediry=data['electron']['diry'],
                        edirz=data['electron']['dirz'], phmult=data['photon']['mult'], phenergy=data['photon']['energy'],
                        phdiry=data['photon']['diry'], phdirz=data['photon']['dirz'], other_keys=np.array(list(data['other'].keys())),
                        other_vals=np.array(list(data['other'].values())))
    
    print("Data successfully written to file cosmics_primary_particles.npz!")