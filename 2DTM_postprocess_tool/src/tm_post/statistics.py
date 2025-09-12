import numpy as np
import math
from scipy.special import erfinv

def calculate_probit(x_):
    # calculate rank and remap to quantile of standard gaussian
    rank_x_ = np.argsort(np.argsort(x_))+1 # use argsort twice to ensure the ranking is correct
    rank_x_ = (rank_x_ - 0.5) / max(1, len(x_))
    pro_x_ = np.sqrt(2) * erfinv(2 * rank_x_ - 1.0)
    return rank_x_, pro_x_

def estimate_anisotropic_gaussian_for_probit(pro_x1_, pro_x2_):
    pro_lim_ = [-4.5, 4.5]
    # define anisotropic-gaussian
    n_pro_x = 128
    n_pro_y = n_pro_x+1
    pro_x_ = np.linspace(min(pro_lim_), max(pro_lim_), n_pro_x)
    pro_dx = np.mean(np.diff(pro_x_))

    pro_y_ = np.linspace(min(pro_lim_), max(pro_lim_), n_pro_y)
    pro_dy = np.mean(np.diff(pro_y_))

    pro_x__, pro_y__ = np.meshgrid(pro_x_,pro_y_)

    n_r = len(pro_x1_)
    tmp_ = np.array([pro_x1_, pro_x2_])
    Cinv__ = np.matmul(tmp_, tmp_.T)/n_r
    Dinv_,Uinv__ = np.linalg.eig(Cinv__) # eigenvalues not necessarily ordered
    return Dinv_, Uinv__ #C__, g__, pro_x__,pro_y__, pro_dx, pro_dy


def calculate_1q_p_value(ag_x1_, ag_x2_, Dinv_, Uinv__):
    # anisotropic gaussian x1 and x2
    # calculate p-value with a 1-quadrant constraint
    n_r = len(ag_x1_)
    p1q_equ_r_ = np.zeros(n_r)
    tmp_a = max(np.sqrt(Dinv_))
    tmp_b = min(np.sqrt(Dinv_))
    #print("Dinv = [", Dinv_[0], ", ", Dinv_[1], "]")
    #print("first eig-vec = ", Uinv__[:,0])
    #print("second eig-vec = ", Uinv__[:,1])
    if Dinv_[0]<=Dinv_[1]:
        Uinv__ = np.array([Uinv__[:,1], Uinv__[:,0]]).T
    if (Uinv__[1,0] < 0) and (Uinv__[0,0]<0):
        Uinv__[1,0]*=-1
        Uinv__[0,0]*=-1
    tmp_w = math.atan2(Uinv__[1,0], Uinv__[0,0])
    # print("Major axis = "+str(tmp_w*180/np.pi)+" deg \n\n")
    tmp_gamma = math.atan2(1.0, 0.5*np.sin(2*tmp_w)*(tmp_b/tmp_a-tmp_a/tmp_b)) # angular formula only valid in 2-dimensions
    for i in range(0, n_r):
        ag_x1 = ag_x1_[i]
        ag_x2 = ag_x2_[i]
        p1q_equ = 1.0
        if (ag_x1>0) and (ag_x2>0):
            tmp_x_0 = +np.cos(tmp_w)*ag_x1 + np.sin(tmp_w)*ag_x2 
            tmp_x_1 = -np.sin(tmp_w)*ag_x1 + np.cos(tmp_w)*ag_x2 # R*S: [cos(w) -sin(w), sin(w) cos(w)]*[sx 0, 0 sy] rotation matrix * scaling matrix
            tmp_y_0 = tmp_x_0/max(1e-12, tmp_a)
            tmp_y_1 = tmp_x_1/max(1e-12, tmp_b)
            tmp_y_r = np.sqrt(tmp_y_0**2+tmp_y_1**2)
            p1q_equ = np.exp(-tmp_y_r**2/2)*tmp_gamma/(2*np.pi)
        p1q_equ_r_[i] = p1q_equ

    return p1q_equ_r_,-np.log(p1q_equ_r_)

def calculate_3q_p_value(ag_x1_, ag_x2_, Dinv_, Uinv__):
    # anisotropic gaussian x1 and x2
    # calculate p-value with a 1-quadrant constraint
    n_r = len(ag_x1_)
    p1q_equ_r_ = np.zeros(n_r)
    tmp_a = max(np.sqrt(Dinv_))
    tmp_b = min(np.sqrt(Dinv_))
    #print("Dinv = [", Dinv_[0], ", ", Dinv_[1], "]")
    #print("first eig-vec = ", Uinv__[:,0])
    #print("second eig-vec = ", Uinv__[:,1])
    if Dinv_[0]<=Dinv_[1]:
        Uinv__ = np.array([Uinv__[:,1], Uinv__[:,0]]).T
    if (Uinv__[1,0] < 0) and (Uinv__[0,0]<0):
        Uinv__[1,0]*=-1
        Uinv__[0,0]*=-1
    tmp_w = math.atan2(Uinv__[1,0], Uinv__[0,0])
    # print("Major axis = "+str(tmp_w*180/np.pi)+" deg \n\n")
    tmp_gamma = math.atan2(1.0, 0.5*np.sin(2*tmp_w)*(tmp_b/tmp_a-tmp_a/tmp_b)) # angular formula only valid in 2-dimensions
    for i in range(0, n_r):
        ag_x1 = ag_x1_[i]
        ag_x2 = ag_x2_[i]
        p1q_equ = 1.0
        if (ag_x1>0) or (ag_x2>0):
            tmp_x_0 = +np.cos(tmp_w)*ag_x1 + np.sin(tmp_w)*ag_x2 
            tmp_x_1 = -np.sin(tmp_w)*ag_x1 + np.cos(tmp_w)*ag_x2 # R*S: [cos(w) -sin(w), sin(w) cos(w)]*[sx 0, 0 sy] rotation matrix * scaling matrix
            tmp_y_0 = tmp_x_0/max(1e-12, tmp_a)
            tmp_y_1 = tmp_x_1/max(1e-12, tmp_b)
            tmp_y_r = np.sqrt(tmp_y_0**2+tmp_y_1**2)
            p1q_equ = np.exp(-tmp_y_r**2/2)*tmp_gamma/(2*np.pi)
        p1q_equ_r_[i] = p1q_equ

    return p1q_equ_r_,-np.log(p1q_equ_r_)

def calculate_2dtm_pval(zscores, snrs, q=1):
    """
    Calculate the 2D p-value for a given z-score and SNR using anisotropic Gaussian distribution.
    Args:
        zscore (numpy.ndarray): Array of z-scores.
        snr (numpy.ndarray): Array of SNR values.
    Returns:
        numpy.ndarray: Array of (negative log) p-values.
    """
    x1_ = zscores
    x2_ = snrs
    _, pro_x1_ = calculate_probit(x1_)
    _, pro_x2_ = calculate_probit(x2_)

    Dinv_, Uinv__ = estimate_anisotropic_gaussian_for_probit(pro_x1_, pro_x2_)
    if q==1:
        pval,neg_log_p1q_equ_pro_ = calculate_1q_p_value(pro_x1_, pro_x2_, Dinv_, Uinv__)
    if q==3:
        pval,neg_log_p1q_equ_pro_ = calculate_3q_p_value(pro_x1_, pro_x2_, Dinv_, Uinv__)
    return neg_log_p1q_equ_pro_