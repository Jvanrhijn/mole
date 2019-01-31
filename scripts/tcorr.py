#! /usr/bin/env python3
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from statfor import correlation


cmd = "cargo run --release {}"


def get_tcor_and_acceptance(box_side):
    process = subprocess.Popen(cmd.format(box_side).split(), 
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, err = process.communicate()

    data = np.array([float(x) for x in out.decode("ascii").split("\n")[:-1]])
    acceptance = data[-1]
    mean = np.mean(data[:-1])
    var = np.var(data[:-1])

    return correlation(data[:-1], mean, var)[0], acceptance


sides = np.linspace(0.02, 15, 50)
tcorr = []
acceptance = []

for side in sides:
    tc, ac = get_tcor_and_acceptance(side)
    tcorr.append(tc)
    acceptance.append(ac)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax2.plot(sides, acceptance, 'r')
ax1.plot(sides, tcorr, 'b')

ax1.set_xlabel("Box side")
ax1.set_ylabel("Tcorr", color='b')
ax1.tick_params('y', colors='b')


ax2.set_ylabel("Acceptance rate", color='r')
ax2.tick_params('y', colors='r')

plt.grid(True)
plt.show()

