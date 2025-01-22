import numpy as np

# ---------------------------------------------------------------------------- #
#                   Utils used by the urban tree preprocessor                  #
# ---------------------------------------------------------------------------- #


def generate_cylinder_points(start, axis, length, radius, num_points):
    """
    Generate points on the circumference of a cylinder (excluding top and bottom caps).

    Parameters:
        start (np.array): The starting point (x, y, z) of the cylinder.
        axis (np.array): The axis vector (normalized) of the cylinder.
        length (float): The length of the cylinder.
        radius (float): The radius of the cylinder.
        num_points (int): The number of points to generate.

    Returns:
        np.array: A Nx3 array of points sampled on the cylinder's surface.
    """
    axis = axis / np.linalg.norm(axis)  # Ensure axis is a unit vector
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # Angle around the circumference
    heights = np.linspace(0, length, max(2, num_points // 10))  # Heights along the cylinder axis

    points = []
    for h in heights:
        circle_center = start + h * axis  # Move along the cylinder's axis
        for angle in theta:
            offset = radius * np.array([np.cos(angle), np.sin(angle), 0])  # Offset in the plane
            # Rotate offset to align with the cylinder's axis
            rot_matrix = get_rotation_matrix(axis)
            rotated_offset = rot_matrix @ offset
            point = circle_center + rotated_offset
            points.append(point)

    return np.array(points)


def get_rotation_matrix(axis):
    """
    Compute a rotation matrix to align a cylinder's local axis with the global z-axis.
    """
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, axis)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, axis)

    if s == 0:  # Axis is already aligned with z-axis
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
