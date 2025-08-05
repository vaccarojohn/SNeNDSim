import sys
from processHDF5 import main

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python HDF5batch.py [prefix] [numZeros] [start] [end]. Ex: python HDF5batch.py CosmicFlux_g4_ 4 0 199")
    else:
        argv = []
        for i in range(int(sys.argv[3]), int(sys.argv[4]) + 1):
            argv.append(sys.argv[1] + format(i, '0' + str(sys.argv[2])) + '.h5')
            
        main(argv)