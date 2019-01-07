"""Read a stream of data from a file, perform a statistical analysis"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


MAX_STEPS = 200


def read_data(fpath):
    """Read data from file located at fpath"""
    return np.array(list(float(line) for line in open(fpath, "r")))


def correlation(data, mean, var):
    """Calculate correlation time, effective N, error"""
    nsteps = len(data)
    tcorr = 1
    for i in range(1, min(MAX_STEPS, nsteps-1)+1):
        corr = ((data[:nsteps-i] - mean)*(data[i:] - mean))[:nsteps-i].sum()
        corr /= (nsteps-i)*var
        tcorr += 2*corr*(0 if corr < 0 else 1)
    tcorr = max(1, tcorr)
    return tcorr, nsteps/tcorr, math.sqrt(var*tcorr/nsteps)


if __name__ == "__main__":
    SERIES = read_data(sys.argv[1])

    MEAN = SERIES.mean()
    VAR = SERIES.var(ddof=1)

    TCORR, NEFF, SIGMA = correlation(SERIES, MEAN, VAR)

    # Print output
    print(f"average\t\t{MEAN:.10f}")
    print(f"variance\t{VAR:.10f}")
    print(f"t corr\t\t{TCORR:.10f}")
    print(f"n eff\t\t{NEFF:.10f}")
    print(f"sigma\t\t{SIGMA:.10f}")
