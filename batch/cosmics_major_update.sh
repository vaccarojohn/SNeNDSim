directory="/sdf/data/neutrino/yuntse/coherent/SNeNDSens/g4/Cosmics"
for subdir in "$directory"/*; do
	cp "$subdir"/*.h5 /sdf/data/neutrino/jvaccaro/SNeNDSens/edepsim/Cosmics_unprocessed
done
