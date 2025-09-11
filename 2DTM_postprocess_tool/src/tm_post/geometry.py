import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_rotation(psi, theta, phi, degrees=True):
    """
    Convert Euler angles (psi, theta, phi) to a Rotation object using the ZYZ convention.
    The rotations are applied in the order: 
      1. Rotation about the Z-axis by psi,
      2. Rotation about the Y-axis by theta,
      3. Rotation about the Z-axis by phi.
    
    This convention aligns with the angular annotations used in RELION and cisTEM.
    """
    return R.from_euler('ZYZ', [psi, theta, phi], degrees=degrees)

def euler_to_matrix(psi, theta, phi, degrees=True):
    return R.from_euler('ZYZ', [psi, theta, phi], degrees=degrees).as_matrix()

def geodesic_distance(ref_rot, pixel_rot):
    """
    Compute the geodesic distance (in radians) between two rotations.
    
    Parameters:
    - ref_rot: the reference Rotation object.
    - pixel_rot: the Rotation object of the pixel to compare.
    
    The geodesic distance is the magnitude of the rotation vector
    corresponding to the relative rotation between ref_rot and pixel_rot.
    """
    # Compute the relative rotation from reference to pixel
    relative_rot = ref_rot.inv() * pixel_rot
    # The magnitude of the rotation vector represents the geodesic distance
    angle = np.linalg.norm(relative_rot.as_rotvec())
    return angle


def Rz(x_azimu):
    x_azimu = x_azimu*np.pi/180
    return np.array([[+np.cos(x_azimu), -np.sin(x_azimu), 0], [+np.sin(x_azimu), +np.cos(x_azimu), 0], [0, 0, 1]])

def Ry(x_polar):
    x_polar = x_polar*np.pi/180
    return np.array([[+np.cos(x_polar), 0, +np.sin(x_polar)], [0, 1, 0], [-np.sin(x_polar), 0,  +np.cos(x_polar)]])

def return_euler_err(gt_psi,gt_theta,gt_phi, tm_psi, tm_theta,tm_phi):
    n_gamma_z = 1024 
    gamma_z_ = np.linspace(0,2*np.pi,n_gamma_z+1) 
    gamma_z_ = np.transpose(gamma_z_[0:n_gamma_z])
    ring_k_c_0_ = np.cos(gamma_z_)
    ring_k_c_1_ = np.sin(gamma_z_)
    ring_k_c_2_ = np.zeros(n_gamma_z)
    ring_k_c_3z__ = np.array([ring_k_c_0_,ring_k_c_1_,ring_k_c_2_])
    tmp_ring_est_k_c_3z__ = np.matmul(Rz(tm_phi),Ry(tm_theta))
    tmp_ring_est_k_c_3z__ = np.matmul(tmp_ring_est_k_c_3z__, Rz(tm_psi))
    tmp_ring_est_k_c_3z__ = np.matmul(tmp_ring_est_k_c_3z__, ring_k_c_3z__)
    tmp_ring_tru_k_c_3z__ = np.matmul(Rz(gt_phi),Ry(gt_theta))
    tmp_ring_tru_k_c_3z__ = np.matmul(tmp_ring_tru_k_c_3z__, Rz(gt_psi))
    tmp_ring_tru_k_c_3z__ = np.matmul(tmp_ring_tru_k_c_3z__, ring_k_c_3z__)
    tmp_ring_l2 = np.sqrt(np.sum((tmp_ring_est_k_c_3z__ - tmp_ring_tru_k_c_3z__)**2)*2*np.pi/max(1,n_gamma_z))
    return tmp_ring_l2