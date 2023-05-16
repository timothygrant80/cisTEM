#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import sys
import numpy as np


#Get first argument
filename = sys.argv[1]
with open(filename) as f:
    s = f.read()
    s = s.replace('nan', '0.0')
    data = json.loads(s)

# Create two plots
fig, ((ax1, ax2),( ax3, ax4), (ax5, ax6), (ax7,ax8)) = plt.subplots(4, 2, sharex=False)

# Plot "initial_fit" in ax1
ax1.plot(data['spatial_frequency'], data['initial_fit']['rotational_average_astig'], 'b-')
ax1.plot(data['spatial_frequency'], data['initial_fit']['rotational_average_astig_fit'], 'r-')
ax1.set_title('Initial fit')

# Plot "after_first_estimate" in ax2
ax2.plot(data['spatial_frequency'], data['after_first_estimate']['rotational_average_astig'], 'b-')                     
ax2.plot(data['spatial_frequency'], data['after_first_estimate']['rotational_average_astig_fit'], 'r-')
ax2.set_title(f'After first thickness estimate {data["thickness_estimates"]["initial"]:.2f} A')

# PLot the output of the 1D search in ax 3
all_values = np.array(data['1D_brute_force_search']['all_values']).reshape(-1, 2)
Xs = all_values[:,0].reshape(-1,351)
np.set_printoptions(threshold=sys.maxsize)
print(Xs)
Ys = all_values[:,1].reshape(-1,351)
all_scores = np.array(data['1D_brute_force_search']['all_scores']).reshape(1,-1)
Zs = all_scores.reshape(-1,351)
#Make ax3 a 3D plot
ax3 = fig.add_subplot(4, 2, 3, projection='3d')
ax3.plot_surface(Xs,Ys,Zs, cmap='viridis', edgecolor='none')
#ax3.plot(data['1D_brute_force_search']['all_values'], data['1D_brute_force_search']['all_scores'], 'b-')
ax3.set_title(f'1D search')

# Plot "after_1D_brute_force" in ax4
ax4.plot(data['spatial_frequency'], data['after_1D_brute_force']['rotational_average_astig'], 'b-')                     
ax4.plot(data['spatial_frequency'], data['after_1D_brute_force']['rotational_average_astig_fit'], 'r-')
ax4.set_title(f'After 1D brute force {data["thickness_estimates"]["after_1D_brute_force"]:.2f} A')

# Plot "after_2D_refine" in ax5
ax5.plot(data['spatial_frequency'], data['after_2D_refine']['rotational_average_astig'], 'b-')                     
ax5.plot(data['spatial_frequency'], data['after_2D_refine']['rotational_average_astig_fit'], 'r-')
ax5.set_title(f'After 2D refine {data["thickness_estimates"]["after_2D_refine"]:.2f} A')

# Plot "after_2D_refine" in ax5
ax7.plot(data['spatial_frequency'], data['frc']['renormalized_spectrum'], 'b-')    
ax7.plot(data['spatial_frequency'], data['frc']['fit'], 'r-')

#ax5.set_title(f'After 2D refine {data["thickness_estimates"]["after_2D_refine"]:.2f} A')
ax8.plot(data['spatial_frequency'], data['frc']['number_of_extrema_profile'], 'b-')   
ax8.plot(data['spatial_frequency'], data['frc']['fit'], 'r-')


plt.show()
