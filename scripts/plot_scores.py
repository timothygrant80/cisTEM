#!/usr/bin/python
#
# Alexis Rohou, May 2018
#
# https://docs.python.org/3/library/sqlite3.html
#
import sys
import sqlite3
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Database connection
conn = sqlite3.connect(str(sys.argv[1]))
c = conn.cursor()

# Fetch defocus1 and fit resolution
# REFINEMENT_RESULT_1_1
refinement_package_id = sys.argv[2]
refinement_id = sys.argv[3]
class_id = sys.argv[4]
c.execute("SELECT refinement_package_contained_particles_{}.parent_image_asset_id,refinement_result_{}_{}.defocus1,refinement_result_{}_{}.score FROM refinement_result_{}_{} INNER JOIN refinement_package_contained_particles_{} on refinement_result_{}_{}.position_in_stack=refinement_package_contained_particles_{}.position_in_stack;".format(
    refinement_package_id,refinement_id,class_id,refinement_id,class_id,refinement_id,class_id,refinement_package_id,refinement_id,class_id,refinement_package_id))
scores_fetched = c.fetchall()

# Close the database connection
conn.close()


[image_id,defocus1,scores] =  map(list, zip(*scores_fetched))

plt.subplot(211)
num_bins = 100
n, bins, patches = plt.hist(scores, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Score')
plt.ylabel('Number of particles')
plt.title('Distribution of scores')

plt.subplot(212)
plt.xlim(1000,1050)
plt.scatter(image_id,scores,alpha=0.1)
plt.xlabel('Image id')
plt.ylabel('Score')


plt.show()
