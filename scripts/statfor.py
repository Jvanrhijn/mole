#!/usr/bin/env python3
"""Read a stream of data from a file, perform a statistical analysis"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


MAX_STEPS = 200


def int_width(num):
    """Return decimal with of integer number"""
    return math.floor(math.log10(num))

def read_data(fpath):
    """Read data from file located at fpath"""
    return np.array(list(float(line) for line in open(fpath, "r")))


def correlation(data, mean, var):
    """Calculate correlation time, effective N, error"""
    nsteps = len(data)
    tcorr = 1
    # needed for formatting
    max_i = min(MAX_STEPS, nsteps -1)
    max_width = int_width(max_i)
    f = 1
    with open("corr.out", "w") as file:
        for i in range(1, max_i + 1):
            corr = ((data[:nsteps-i] - mean)*(data[i:] - mean))[:nsteps-i].sum()
            corr /= (nsteps-i)*var
            if corr < 0:
                f = 0
            tcorr += 2*corr*f
            file.write(f"{i}" +" "*(max_width-int_width(i)+4) + f"{corr:.10e}\n")
        tcorr = max(1, tcorr)

    return tcorr, nsteps/tcorr, math.sqrt(var*tcorr/nsteps)


def blocking(data):
    """Perform blocking analysis"""
    ndata = len(data)
    min_left = 20
    nsizes = 100
    large = int(ndata/min_left)
    step_size = int(max(1, large/nsizes))

    # needed for formatting
    max_width = int_width(large+1)
    
    sizes, errors = [], [] 
    with open("blocking.out", "w") as file:
        for size in range(1, large+1, step_size):
            nblocks = int(ndata/size)
            # block averages
            blocks = np.array_split(data, nblocks)
            averages_blk = np.array(list(map(np.mean, blocks)))
            averages_sq_blk = np.array([ab**2 for ab in averages_blk]).mean()
            # estimated error of mean at this block size
            errors.append((np.sqrt((averages_sq_blk - averages_blk.mean()**2)\
                    /(nblocks-1))))
            sizes.append(size)
            file.write(f"{size}" \
                    +" "*(max_width-int_width(size)+4) + f"{errors[-1]:.10e}\n")

    return np.array(errors), np.array(sizes)


if __name__ == "__main__":
    SERIES = read_data(sys.argv[1])

    MEAN = SERIES.mean()
    VAR = SERIES.var(ddof=1)

    TCORR, NEFF, SIGMA = correlation(SERIES, MEAN, VAR)

    ERRORS, SIZES = blocking(SERIES)
    
    if "-p" in sys.argv:
        plt.plot(SIZES, ERRORS, '.')
        plt.ylabel("Error")
        plt.xlabel("Block size")
        plt.grid()
        plt.show()

    # Print output
    print(f"average\t\t{MEAN:.10f}")
    print(f"variance\t{VAR:.10f}")
    print(f"t corr\t\t{TCORR:.10f}")
    print(f"n eff\t\t{NEFF:.10f}")
    print(f"sigma\t\t{SIGMA:.10f}")
