#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import glob


tiltseries_data = []

for i, ff in enumerate(glob.glob('/nrs/elferich/Tiltseries/Assets/CTF/*tilt.json')):
    with open(ff) as f:
        data = json.load(f)
    tilt = float(ff.split('_')[3])
    # Create two plots
    tiltseries_data.append((tilt, data))
    
tiltseries_data.sort(key=lambda x: x[0])


# One subplot per tilt


for i, (tilt, data) in enumerate(tiltseries_data):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    axis, angle, score =  zip(*data["tilt_axis_and_angle_search"])
    score = np.array(score).reshape(13, 36)
    print(score.shape)
    axis = np.array(axis).reshape(13, 36)
    angle = np.array(angle).reshape(13, 36)

    # Plot "initial_fit" in ax1
    ax.plot_surface(axis, angle, score, cmap='viridis', edgecolor='none')
    ax.set_title(f"Tilt {tilt:.1f}")


    plt.show()
