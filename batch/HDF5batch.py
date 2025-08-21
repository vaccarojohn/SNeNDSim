import sys
from processHDF5 import main

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python HDF5batch.py [infolder] [outfolder] [filePrefix] [start] [end].")
    else:
        argv = []
        for i in range(int(sys.argv[4]), int(sys.argv[5]) + 1):
            argv.append((sys.argv[1] + '/' + sys.argv[3] + format(i, "04") + '.h5', sys.argv[2] + '/' + sys.argv[3] + format(i, "04") + '-processed.h5'))
            
        main(argv)