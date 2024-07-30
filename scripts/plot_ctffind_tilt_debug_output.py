#!/usr/bin/env python
# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import glob


tiltseries_data = []

# for i, ff in enumerate(glob.glob('/nrs/elferich/Tiltseries/Assets/CTF/*tilt.json')):
#13: 27,72
#14: search step is 1 (81,360),the best is (81,61)degrees
#19: box size 128, defocus refine locked. best angle and tilt is 54,87, range 20,10
#22: box size 512, defocus refine locked. best angle and tilt is 61,81, range 20,10
#23: box size 512, defocus refine locked. best angle and tilt is 61,81, range 40,20
# when box size 256, and defocus refine locked, the refinement enters dead loop
# then change the initial point, line 366 changed.
#33: box size 256
#41: box size 640, refine tilt and angle. best is 61.71,81.76
#42: box size 512, low reso=35. refine tilt, angle, and defocus, best is 54.42, 89.89.
#43: box size 512, low reso=25, refine tilt, angle, and defocus, best is 54.42, 89.89.
#46: low reso 15, resample pixel size 1.4, defocus is not correctly found
#47: low reso 15, resample pixel size 2.8
#48: low reso 15, high reso 10.0, pixel size recovered to 1.4.
#49: low reso 40, high reso 10.0, best is 54.42, 89.89.
#50: low reso 30, high reso 5.0 best is 54.42, 89.89
#51: box size 512, croped to 8192*8192, reso the same as above, best is 56.31, 89.91
#52: box size 128, croped to 8192*8192, reso the same as above, best is 54.66, 88.71
#53: box size 256, croped to 8192*8192, reso the same as above, best is 57.08, 87.57
#54: box size 256, croped to 8192*8192, fixed defocus, best 55.68,83.64. but score is positive, which is weird
#55: (0) box size 256, croped to 4096*4096, fixed defocus, best 53.96, 81.36
#56: box size 256, croped to 4096*4096, defocus not fixed, best 60.34, 86.97
#57-59: cropped to 4096, crop center is 4096. the defocus is not correct
#61: cropped to 4096, crop center is 6624, best: 59.45, 96.61
#64: at the refinement, tilt angle, axis, and defocus uses the result from ctffind3. best 56.08, 91.20. score is around -0.058754
#66: the original code with box size 256, best is 53.32, 89.46. score around -0.060839
#67: increase the ranges[3]=2000, best is 53.32, 89.46. score around -0.060839
#the above test shows that, with different initial data, the program converge to different point.
#68: box=128, n_section=5, n_step=6. best 56.68, 88.33
#69: box=128, n_section=3, n_step=6. best 51.11, 86.89
#70: box=128, n_section=5, n_step=8. best 56.77, 88.09
#71: box=128, n_section=7, n_step=10, best 47.79, 90.85
#72: tile size force to 128, n_section=21, n_steps=2. best 61.11, 87.60
#73: for image 550089, for the same condition as 72, best 60.65, 88.18
for i, ff in enumerate(glob.glob('/data/lingli/CTFTiltFit/Crystal_Tilt/Crystal_CTF/Crystal_CTF_fitnode/Assets/CTF/*3*tilt.json')):

    with open(ff) as f:
        data = json.load(f)
    # tilt = float(ff.split('_')[3])
    tilt = 0
    # Create two plots
    tiltseries_data.append((tilt, data))
    
tiltseries_data.sort(key=lambda x: x[0])


# One subplot per tilt


for i, (tilt, data) in enumerate(tiltseries_data):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    axis, angle, score =  zip(*data["tilt_axis_and_angle_search"])
    score = np.array(score).reshape(17, 36)
    print(score.shape)
    axis = np.array(axis).reshape(17, 36)
    angle = np.array(angle).reshape(17, 36)
#    score = score[:13,:]
#    axis = axis[:13,:]
#    angle = angle[:13,:]

    # Plot "initial_fit" in ax1
    ax.plot_surface(axis, angle, score, cmap='viridis', edgecolor='none')
    # ax.plot_surface(axis[50:70,70:100], angle[50:70,70:100], score[50:70,70:100], cmap='viridis', edgecolor='none')
    ax.set_title(f"Tilt {tilt:.1f}")
    # ax.set_zlim(0,0.01)

    plt.show()

# %%
