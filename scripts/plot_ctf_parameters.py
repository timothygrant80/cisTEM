#!/usr/bin/python
#
# Alexis Rohou, March 2018
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
c.execute('SELECT defocus1,detected_ring_resolution FROM estimated_ctf_parameters;')
def_and_res = c.fetchall()

# Close the database connection
conn.close()

# Get separate defocus and fit resolution lists
[defocus1,fitres] =  map(list, zip(*def_and_res))

plt.subplot(211)
num_bins = 20
n, bins, patches = plt.hist(fitres, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('CTF fit resolution (A)')
plt.ylabel('Number of micrographs')
plt.title('Goodness of CTF fit')

plt.subplot(212)
num_bins = 20
n, bins, patches = plt.hist(defocus1, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Defocus (A)')
plt.ylabel('Number of micrographs')
plt.title('Defocus')



plt.show()
