import numpy as np

# ---------------------------------------------------------------------------- #
#                   Utils used by the urban tree preprocessor                  #
# ---------------------------------------------------------------------------- #


def generate_cylinder_points(start, axis, length, radius, num_points):
    """
    Generate exactly `num_points` points uniformly or randomly across the curved surface
    of a cylinder (excluding top and bottom caps), optimized for performance.

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

    # Randomly sample heights along the cylinder's length
    heights = np.random.uniform(0, length, num_points)

    # Randomly sample angles around the circumference
    theta = np.random.uniform(0, 2 * np.pi, num_points)

    # Compute the circle centers along the axis for all heights
    circle_centers = start + np.outer(heights, axis)

    # Compute the offsets in the local plane
    x_offsets = radius * np.cos(theta)
    y_offsets = radius * np.sin(theta)
    offsets = np.column_stack((x_offsets, y_offsets, np.zeros(num_points)))

    # Rotate offsets to align with the cylinder's axis
    rot_matrix = get_rotation_matrix(axis)
    rotated_offsets = offsets @ rot_matrix.T  # Apply rotation to all offsets

    # Combine circle centers and rotated offsets to get final points
    points = circle_centers + rotated_offsets

    return points


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
