import numpy as np

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int(x1 + x2) / 2, int(y2))

def get_closest_keypoint_index(foot_position, keypoints_flat, indices):
    """
    keypoints_flat: 1D array-like, [x0,y0,x1,y1,…]
    indices: list of keypoint IDs to consider
    Returns: the keypoint ID (from indices) whose (x,y) is nearest to foot_position.
    """
    fx, fy = foot_position
    # ensure it’s a numpy array
    kp = np.asarray(keypoints_flat)
    # compute distances only on the selected indices
    dists = []
    for i in indices:
        xi = kp[2*i]
        yi = kp[2*i + 1]
        dists.append(np.hypot(fx-xi, fy-yi))
    best = indices[int(np.argmin(dists))]
    return best

def get_height_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int(y2 - y1)

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])