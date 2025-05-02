import numpy as np
import cv2



def extract_view_from_panorama(
    panorama: np.ndarray,
    heading: float,
    pitch: float = 0,
    fov: float = 90,
    width: int = 640,
    height: int = 640,
    heading_offset: float = 0,
    vertical_fov: float = None
) -> np.ndarray:
    """
    Extract a view from a panoramic Street View Image. 

    Parameters
    ----------
    panorama
        A numpy array containing the full panorama. 
    heading
        The heading (sideways) angle of the desired view.
    pitch
        The pitch (elevation) angle of the desired view.
    fov
        The field of view.
    width
        Width of the extracted view.
    height
        Height of the extracted view.
    heading_offset
        Usually panoramas start at global north but if they start on another angle an offset value should be provided. 
    vertical_fov
        If the panoramic does not span the whole 180 degrees in the vertical direction that has to be specified to avoid distortions

    Returns
    -------
    view
        Requested view as a 3-channel numpy array.
    """
    
    if(height is None):
        height = compute_height_from_fov(fov, width, aspect_ratio)
    # Apply heading correction
    corrected_heading = (heading - heading_offset) % 360

    h_src, w_src, _ = panorama.shape
    #Intrinsics
    #focal length
    f = 0.5 * width / np.tan(0.5 * fov / 180.0 * np.pi)
    #Center of the camera
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0
    #Camera matrix approximation
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    K_inv = np.linalg.inv(K)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = np.ones_like(x)
    directions = np.stack([x, y, z], axis=-1) @ K_inv.T

    y_axis = np.array([0, 1, 0], dtype=np.float32)
    x_axis = np.array([1, 0, 0], dtype=np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(corrected_heading))
    R2, _ = cv2.Rodrigues((R1 @ x_axis) * np.radians(pitch))
    R = R2 @ R1
    directions = directions @ R.T

    norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    dirs = directions / norm
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    u = (lon / (2 * np.pi) + 0.5) * (w_src - 1)
    # Compute actual vertical FOV based on image shape (in radians) # assumes 360° horizontal coverage

    # Correct vertical mapping
    if vertical_fov is None:
        if np.isclose(width / height, 2.0, atol=0.1):
            vertical_fov_radians = np.pi
        else:
            vertical_fov_radians = np.radians(fov)
    else:
        vertical_fov_radians = np.radians(vertical_fov)
    v = (lat / vertical_fov_radians + 0.5) * (h_src - 1)

    return cv2.remap(
        panorama, u.astype(np.float32), v.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP
    )