#! /usr/bin/env python3
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from statfor import correlation, read_data


cmd = "cargo run --release {}"


def get_tcor_and_acceptance(box_side):
    process = subprocess.Popen(cmd.format(box_side).split(), 
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, err = process.communicate()

    lines = out.decode("ascii").split("\n")[:-1]
    data = np.array([float(line.split()[1]) for line in lines])
    acceptance = float(lines[-1].split()[-1])
    mean = np.mean(data)
    var = np.var(data)

    return correlation(data[:-1], mean, var)[0], acceptance


sides = np.linspace(0.02, 30, 50)
tcorr = []
acceptance = []

for side in sides:
    tc, ac = get_tcor_and_acceptance(side)
    tcorr.append(tc)
    acceptance.append(ac)

fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()

#ax2.plot(sides, acceptance, 'r')
ax1.plot(sides, tcorr, 'b')

ax1.set_xlabel("Box side (atomic units)")
ax1.set_ylabel("Tcorr", color='b')
ax1.tick_params('y', colors='b')


#ax2.set_ylabel("Acceptance rate", color='r')
#ax2.tick_params('y', colors='r')

plt.grid(True)
plt.show()

