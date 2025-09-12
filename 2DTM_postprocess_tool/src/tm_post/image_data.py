from dataclasses import dataclass

@dataclass
class TMImage:
    image_id: int
    filename: str
    
    snr_file: str
    zscore_file: str
    avg_file: str
    sd_file: str
    
    psi_file: str
    theta_file: str
    phi_file: str
    
    defocus_file: str
    pixel_size: float
    defocus1: float
    defocus2: float
    defocus_angle: float

    amp_contrast: float
    voltage: float
    cs: float