#!/usr/bin/env python

import json
import matplotlib.pyplot as plt

with open('debug.json') as f:
    data = json.load(f)

# Create two plots
fig, ((ax1, ax2),( ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=False)

# Plot "initial_fit" in ax1
ax1.plot(data['spatial_frequency'], data['initial_fit']['rotational_average_astig'], 'b-')
ax1.plot(data['spatial_frequency'], data['initial_fit']['rotational_average_astig_fit'], 'r-')
ax1.set_title('Initial fit')

# Plot "after_first_estimate" in ax2
ax2.plot(data['spatial_frequency'], data['after_first_estimate']['rotational_average_astig'], 'b-')                     
ax2.plot(data['spatial_frequency'], data['after_first_estimate']['rotational_average_astig_fit'], 'r-')
ax2.set_title(f'After first thickness estimate {data["thickness_estimates"]["initial"]:.2f} A')

# PLot the output of the 1D search in ax 3
ax3.plot(data['1D_brute_force_search']['all_values'], data['1D_brute_force_search']['all_scores'], 'b-')
ax3.set_title(f'1D search')

# Plot "after_1D_brute_force" in ax4
ax4.plot(data['spatial_frequency'], data['after_1D_brute_force']['rotational_average_astig'], 'b-')                     
ax4.plot(data['spatial_frequency'], data['after_1D_brute_force']['rotational_average_astig_fit'], 'r-')
ax4.set_title(f'After 1D brute force {data["thickness_estimates"]["after_1D_brute_force"]:.2f} A')

# Plot "after_2D_refine" in ax5
ax5.plot(data['spatial_frequency'], data['after_2D_refine']['rotational_average_astig'], 'b-')                     
ax5.plot(data['spatial_frequency'], data['after_2D_refine']['rotational_average_astig_fit'], 'r-')
ax5.set_title(f'After 2D refine {data["thickness_estimates"]["after_2D_refine"]:.2f} A')


plt.show()
