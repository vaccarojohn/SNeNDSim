import numpy as np

categories = ('signal', 'cosmics', 'BRN', 'dirt')
file_dir = 'graph_data'

if __name__ == "__main__":
    for category in categories:
        print("Working on " + category + "...")
    
        # Read NumPy segment and selection data
        metadata = np.load(category + '/' + file_dir + '/' + category + '_metadata.npz')
        segment_data = np.load(category + '/' + file_dir + '/' + category + '_segment_data.npz')
        selection_data = np.load(category + '/' + file_dir + '/' + category + '_selection_data.npz')
        vertex_data = np.load(category + '/' + file_dir + '/' + category + '_vertex_data.npz')

        m = 0
        n = 0
        visible_events = []
        for i, (lcte, lctl, pmaxe) in enumerate(np.column_stack((segment_data['lctes'], segment_data['lctls'], selection_data['pmaxe']))):
            if lcte < 5:
                #print("Discovered invisible " + category + " event. Event " + str(i) + " has a leading charged trajectory of " + str(lcte) + ".")
                m += 1
            elif lctl < 2:
                #print("Discovered invisible " + category + " event. Event " + str(i) + " has a leading charged trajectory length of " + str(lctl) + ".")
                m += 1
            elif pmaxe < 20:
                selection_data['pmaxe'][i] = 0
                visible_events.append(i)
                n += 1
            else:
                visible_events.append(i)

        print("Total invisible " + category + " events: " + str(m))
        print("Total invisible " + category + " protons: " + str(n))

        print("Writing to output...")
        visible_events_np = np.array(visible_events)
        np.savez_compressed(category + '/' + file_dir + '/' + category + '_metadata_xgboost.npz', event_ids=metadata['event_ids'][visible_events_np],
                            file_ids=metadata['file_ids'][visible_events_np])
    
        np.savez_compressed(category + '/' + file_dir + '/' + category + '_segment_data_xgboost.npz', fenergy=segment_data['fenergy'][visible_events_np],
                            senergy=segment_data['senergy'][visible_events_np], aenergy=segment_data['aenergy'][visible_events_np], 
                            pfenergy=segment_data['pfenergy'][visible_events_np], psenergy=segment_data['psenergy'][visible_events_np],
                            prenergies=segment_data['prenergies'][visible_events_np], prlengths=segment_data['prlengths'][visible_events_np],
                            lctes=segment_data['lctes'][visible_events_np], lctls=segment_data['lctls'][visible_events_np],
                            lengths=segment_data['lengths'][visible_events_np])
    
        np.savez_compressed(category + '/' + file_dir + '/' + category + '_selection_data_xgboost.npz', pmaxe=selection_data['pmaxe'][visible_events_np],
                            crttop=selection_data['crttop'][visible_events_np], crtbottom=selection_data['crtbottom'][visible_events_np],
                            crtleft=selection_data['crtleft'][visible_events_np], crtright=selection_data['crtright'][visible_events_np],
                            crtfront=selection_data['crtfront'][visible_events_np], crtback=selection_data['crtback'][visible_events_np],
                            crt=selection_data['crt'][visible_events_np], light=selection_data['light'][visible_events_np], 
                            tmin=selection_data['tmin'][visible_events_np], tmax=selection_data['tmax'][visible_events_np], 
                            trms=selection_data['trms'][visible_events_np], tseg=selection_data['tseg'][visible_events_np],
                            tdiff=selection_data['tdiff'][visible_events_np], pangle=selection_data['pangle'][visible_events_np],
                            pangle2=selection_data['pangle2'][visible_events_np])

        np.savez_compressed(category + '/' + file_dir + '/' + category + '_vertex_data_xgboost.npz', e01cm=vertex_data['e01cm'][visible_events_np],
                           pe01cm=vertex_data['pe01cm'][visible_events_np], e05cm=vertex_data['e05cm'][visible_events_np], 
                           pe05cm=vertex_data['pe05cm'][visible_events_np], e1cm=vertex_data['e1cm'][visible_events_np], 
                           pe1cm=vertex_data['pe1cm'][visible_events_np], e2cm=vertex_data['e2cm'][visible_events_np], 
                           pe2cm=vertex_data['pe2cm'][visible_events_np], e3cm=vertex_data['e3cm'][visible_events_np], 
                           pe3cm=vertex_data['pe3cm'][visible_events_np])


        