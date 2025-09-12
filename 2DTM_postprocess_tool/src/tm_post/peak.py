from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class Peak:
    image_id: int
    image_name: str
    x: float # in pixel
    y: float # in pixel
    delta_defocus: float
    psi: float
    theta: float
    phi: float
    snr: float
    zscore: float
    pval: float
    avg: float
    sd: float

    #def get_peak_coordinates(self):
    #    return np.array([self.x * self.pixel_size, self.y * self.pixel_size])
    
    def convert_to_rotation_matrix(self):
        # Create rotation matrix from Euler angles (psi, theta, phi)
        return R.from_euler('ZYZ', [self.psi, self.theta, self.phi], degrees=True)
    
    def convert_to_starfile_row(self):
        return [
            "", # empty column
            round(self.psi, 1),
            round(self.theta, 1),
            round(self.phi, 1),
            round(self.defocus1, 1),
            round(self.defocus2, 1),
            0.0,  # ANGAST placeholder
            round(self.pixel_size, 3),
            200.0,  # microscope voltage
            2.7,    # Cs
            0.1,    # Amp contrast
            0.0, 0.0,  # beam tilt X/Y
            0.0, 0.0,  # image shift X/Y
            f"'{self.image_name}'",
            round(self.x, 2),
            round(self.y, 2),
            round(self.pval, 2),
            round(self.zscore, 2),
            round(self.snr, 2)
        ]
    