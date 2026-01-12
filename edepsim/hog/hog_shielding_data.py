import h5py
import numpy as np
from helper_functions import get_length_in_active_volume, get_length_in_signal_volume, get_length_in_fiducial_volume, get_length_in_cosmic_ray_taggers

infile_dir = ['/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/HOG', '/sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/HOG_noHOGPbPipe']
outfile_dir = 'graph_data'

if __name__ == "__main__":
    for k in range(2):
        data_hogcrttop = []
        data_hogcrtbottom = []
        data_hogcrtleft = []
        data_hogcrtright = []
        data_hogcrtfront = []
        data_hogcrtback = []
        data_hogcrt = []
        data_ahog = []
        data_shog = []
        data_fhog = []
        data_nhog = []
        
        for i in range(10):
            print("Loading file " + str(i + 1) + "/10...")
            f = h5py.File(infile_dir[k] + '/HOG_g4_' + format(i, "04") + '.h5', 'r')
    
            event_id = f['segments'][0]['event_id']
            temp_hogcrttop = 0
            temp_hogcrtbottom = 0
            temp_hogcrtleft = 0
            temp_hogcrtright = 0
            temp_hogcrtfront = 0
            temp_hogcrtback = 0
            temp_hogcrt = 0
            temp_ahog = 0
            temp_shog = 0
            temp_fhog = 0
            temp_nhog = []
    
            for seg in f['segments']:
                if seg['event_id'] > event_id:
                    data_hogcrttop.append(temp_hogcrttop)
                    data_hogcrtbottom.append(temp_hogcrtbottom)
                    data_hogcrtleft.append(temp_hogcrtleft)
                    data_hogcrtright.append(temp_hogcrtright)
                    data_hogcrtfront.append(temp_hogcrtfront)
                    data_hogcrtback.append(temp_hogcrtback)
                    data_hogcrt.append(temp_hogcrttop + temp_hogcrtbottom + temp_hogcrtleft + temp_hogcrtright + temp_hogcrtfront + temp_hogcrtback)
                    data_ahog.append(temp_ahog)
                    data_shog.append(temp_shog)
                    data_fhog.append(temp_fhog)
                    data_nhog.append(len(temp_nhog))
                    
                    temp_hogcrttop = 0
                    temp_hogcrtbottom = 0
                    temp_hogcrtleft = 0
                    temp_hogcrtright = 0
                    temp_hogcrtfront = 0
                    temp_hogcrtback = 0
                    temp_hogcrt = 0
                    temp_ahog = 0
                    temp_shog = 0
                    temp_fhog = 0
                    temp_nhog = []
                    
                    event_id = seg['event_id']
    
                in_active = get_length_in_active_volume(seg['x_start'], seg['x_end'], 
                                                        seg['y_start'], seg['y_end'], 
                                                        seg['z_start'], seg['z_end'])
                
                in_signal = get_length_in_signal_volume(seg['x_start'], seg['x_end'], 
                                                        seg['y_start'], seg['y_end'], 
                                                        seg['z_start'], seg['z_end'])
                
                in_fiducial = get_length_in_fiducial_volume(seg['x_start'], seg['x_end'], 
                                                            seg['y_start'], seg['y_end'], 
                                                            seg['z_start'], seg['z_end'])
                
                crt_dist = get_length_in_cosmic_ray_taggers(seg['x_start'], seg['x_end'],
                                                            seg['y_start'], seg['y_end'],
                                                            seg['z_start'], seg['z_end'])
                
                temp_hogcrttop += seg['dEdx'] * crt_dist[0]
                temp_hogcrtbottom += seg['dEdx'] * crt_dist[1]
                temp_hogcrtleft += seg['dEdx'] * crt_dist[2]
                temp_hogcrtright += seg['dEdx'] * crt_dist[3]
                temp_hogcrtfront += seg['dEdx'] * crt_dist[4]
                temp_hogcrtback += seg['dEdx'] * crt_dist[5]
                temp_ahog += seg['dEdx'] * in_active
                temp_shog += seg['dEdx'] * in_signal
                temp_fhog += seg['dEdx'] * in_fiducial
    
                if seg['vertex_id'] not in temp_nhog and in_active != 0:
                    temp_nhog.append(seg['vertex_id'])
    
            data_hogcrttop.append(temp_hogcrttop)
            data_hogcrtbottom.append(temp_hogcrtbottom)
            data_hogcrtleft.append(temp_hogcrtleft)
            data_hogcrtright.append(temp_hogcrtright)
            data_hogcrtfront.append(temp_hogcrtfront)
            data_hogcrtback.append(temp_hogcrtback)
            data_hogcrt.append(temp_hogcrttop + temp_hogcrtbottom + temp_hogcrtleft + temp_hogcrtright + temp_hogcrtfront + temp_hogcrtback)
            data_ahog.append(temp_ahog)
            data_shog.append(temp_shog)
            data_fhog.append(temp_fhog)
            data_nhog.append(len(temp_nhog))
    
            f.close()
                
        print("Writing to output...")

        if k == 0:
            np.savez_compressed(outfile_dir + '/hog_shielded_data.npz', crttop=data_hogcrttop, crtbottom=data_hogcrtbottom, crtleft=data_hogcrtleft,
                                crtright=data_hogcrtright, crtfront=data_hogcrtfront, crtback=data_hogcrtback, crt=data_hogcrt, ahog=data_ahog,
                                shog=data_shog, fhog=data_fhog, nhog=data_nhog)
            print("Data successfully written to file hog_shielded_data.npz!")
        else:
            np.savez_compressed(outfile_dir + '/hog_unshielded_data.npz', crttop=data_hogcrttop, crtbottom=data_hogcrtbottom, crtleft=data_hogcrtleft,
                                crtright=data_hogcrtright, crtfront=data_hogcrtfront, crtback=data_hogcrtback, crt=data_hogcrt, ahog=data_ahog,
                                shog=data_shog, fhog=data_fhog, nhog=data_nhog)
            print("Data successfully written to file hog_unshielded_data.npz!")