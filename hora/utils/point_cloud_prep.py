# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np


def sample_cylinder(h, num_points=100, num_circle_points=15, side_points=70):
    """Sample points from the surface of a unit-radius cylinder.

    Args:
        h: height of the cylinder
        num_points: total number of points to sample
        num_circle_points: number of points on each cap (top and bottom)
        side_points: number of points on the lateral surface
    Returns:
        pcs: (num_points, 3) numpy array of sampled surface points
    """
    assert num_points == num_circle_points * 2 + side_points
    pcs = np.zeros((num_points, 3))
    # sample points from top surface
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[:num_circle_points, 0] = r * np.cos(theta) * 0.5
    pcs[:num_circle_points, 1] = r * np.sin(theta) * 0.5
    pcs[:num_circle_points, 2] = 0.5 * h
    # sample points from bottom surface
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[num_circle_points:num_circle_points * 2, 0] = r * np.cos(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 1] = r * np.sin(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 2] = -0.5 * h
    # sample points from the lateral surface
    vec = np.random.random((side_points, 2)) - 0.5
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    vec *= 0.5
    pcs[num_circle_points * 2:, :2] = vec
    pcs[num_circle_points * 2:, 2] = h * (np.random.random(side_points) - 0.5)
    return pcs
