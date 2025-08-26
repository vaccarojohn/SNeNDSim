import sys
import h5py
import numpy as np
from helper_functions import get_length_in_active_volume

# Output array datatypes
segments_dtype = np.dtype([("event_id","u4"), ("vertex_id", "u8"), ("file_vertex_id", "u8"), ("segment_id", "u4"),
                           ("z_end", "f4"),("traj_id", "i4"), ("file_traj_id", "u4"), ("tran_diff", "f4"),
                           ("z_start", "f4"), ("x_end", "f4"),
                           ("y_end", "f4"), ("n_electrons", "u4"),
                           ("pdg_id", "i4"), ("x_start", "f4"),
                           ("y_start", "f4"), ("t_start", "f4"),
                           ("t0_start", "f8"), ("t0_end", "f8"), ("t0", "f8"),
                           ("dx", "f4"), ("long_diff", "f4"),
                           ("pixel_plane", "i4"), ("t_end", "f4"),
                           ("dEdx", "f4"), ("dE2dx", "f4"), ("dE", "f4"), 
                           ("dE2", "f4"), ("t", "f4"), ("y", "f4"), ("x", "f4"), 
                           ("z", "f4"), ("n_photons","f4")], align=True)

trajectories_dtype = np.dtype([("event_id","u4"), ("vertex_id", "u8"), ("file_vertex_id", "u8"),
                               ("traj_id", "i4"), ("file_traj_id", "u4"), ("parent_id", "i4"), ("primary", "?"),
                               ("E_start", "f4"), ("pxyz_start", "f4", (3,)),
                               ("xyz_start", "f4", (3,)), ("t_start", "f8"),
                               ("E_end", "f4"), ("pxyz_end", "f4", (3,)),
                               ("xyz_end", "f4", (3,)), ("t_end", "f8"),
                               ("pdg_id", "i4"), ("start_process", "i4"),
                               ("start_subprocess", "i4"), ("end_process", "i4"),
                               ("end_subprocess", "i4"), ("dist_travel", "f4")], align=True)

vertices_dtype = np.dtype([("event_id","u4"), ("vertex_id","u8"), ("file_vertex_id", "u8"),
                           ("x_vert","f4"), ("y_vert","f4"), ("z_vert","f4"),
                           ("t_vert","f4"), ("t_event","f4")], align=True)

# Prep HDF5 file for writing
def initHDF5File(output_file):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('trajectories', (0,), dtype=trajectories_dtype, maxshape=(None,))
        f.create_dataset('segments', (0,), dtype=segments_dtype, maxshape=(None,))
        f.create_dataset('vertices', (0,), dtype=vertices_dtype, maxshape=(None,))

# Resize HDF5 file and save output arrays
def updateHDF5File(output_file, trajectories, segments, vertices):
    if any([len(trajectories), len(segments), len(vertices)]):
        with h5py.File(output_file, 'a') as f:
            if len(trajectories):
                ntraj = len(f['trajectories'])
                f['trajectories'].resize((ntraj+len(trajectories),))
                f['trajectories'][ntraj:] = trajectories

            if len(segments):
                nseg = len(f['segments'])
                f['segments'].resize((nseg+len(segments),))
                f['segments'][nseg:] = segments

            if len(vertices):
                nvert = len(f['vertices'])
                f['vertices'].resize((nvert+len(vertices),))
                f['vertices'][nvert:] = vertices

def main(argv):
    for infile, outfile in argv:
        print("Opening file " + infile + "...")
        f = h5py.File(infile, 'r')
        initHDF5File(outfile)
    
        data_event_ids = []
        event_id = f['segments'][0]['event_id']
        temp_inDet = False

        print("Looking for events in TPC...")
        for seg in f['segments']:
            if seg['event_id'] > event_id:
                if temp_inDet:
                    data_event_ids.append(event_id)
                    temp_inDet = False

                event_id = seg['event_id']
                    
            if not temp_inDet and get_length_in_active_volume(seg['x_start'], seg['x_end'], seg['y_start'], seg['y_end'], seg['z_start'], seg['z_end']) != 0:
                temp_inDet = True

        if temp_inDet:
            data_event_ids.append(event_id)

        vertices = f['vertices'][np.isin(f['vertices']['event_id'], data_event_ids)]
        trajectories = f['trajectories'][np.isin(f['trajectories']['event_id'], data_event_ids)]
        segments = f['segments'][np.isin(f['segments']['event_id'], data_event_ids)]
            
        print("Writing to output...")
        updateHDF5File(outfile, trajectories, segments, vertices)
        print("Successfully written to output file " + outfile + "!")
    
    