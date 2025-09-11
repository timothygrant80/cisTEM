import mrcfile
import numpy as np

def read_mrc_file(filename):
  with mrcfile.open(filename) as mrc:
    df = np.squeeze(mrc.data)
    
  return df