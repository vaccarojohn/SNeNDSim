import sys
from dumpTreeNew import dump

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python dumpTreeNewBatch.py [infolder] [outfolder] [filePrefix] [start] [end].")
    else:
        files = []
        for i in range(int(sys.argv[4]), int(sys.argv[5]) + 1):
            infile = sys.argv[1] + '/' + sys.argv[3] + format(i, "04") + '.root'
            outfile = sys.argv[2] + '/' + sys.argv[3] + format(i, "04") + '.h5'

            print("Dumping file " + infile + "...")
            dump(infile, outfile, True)
            print("Successfully dumped to file " + outfile + "!")