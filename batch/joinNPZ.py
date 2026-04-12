import sys
import numpy as np

def main(infiles, outfile):
    loaded_file = np.load(infiles[0])
    data = {}
    
    for key in loaded_file.files:
        data[key] = loaded_file[key]
        
    for infile in infiles[1:]:
        new_loaded_file = np.load(infile)
        for key in loaded_file.files:
            data[key] = np.concatenate((data[key], new_loaded_file[key]))

    np.savez_compressed(outfile, **data)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python joinNPZ.py [outfile] [infolder] [file1] [file2] ...")
    else:
        infiles = []
        for i in range(len(sys.argv) - 3):
            infiles.append(sys.argv[2] + '/' + sys.argv[i + 3])

        main(infiles, sys.argv[1])