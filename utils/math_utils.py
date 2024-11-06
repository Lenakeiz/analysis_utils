import numpy as np

def rotate_vector(direction, angle, is_clockwise):
    angle_rad = np.deg2rad(angle)
    if is_clockwise == True:
        angle_rad = -angle_rad

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_direction = np.dot(rotation_matrix, direction)
    return rotated_direction

def normalize_vector(vector):
    # Normalize the vector
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.array([0, 0])
    direction = vector / magnitude
    
    return direction

def calculate_distance(vector1 : np.array, vector2 : np.array=None):
    if vector2 is not None:
        # Calculate the distance between two points
        return np.linalg.norm(np.array(vector1) - np.array(vector2))
    else:
        # Calculate the length of a vector
        return np.linalg.norm(vector1)

def calculate_unsigned_angle(v1, v2, isClockwise : bool):
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    dot_product = np.dot(v1,v2)
    # Clamp dot_product to avoid numerical issues with arccos
    dot_product = max(min(dot_product, 1.0), -1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    cross_product = np.cross(v1, v2)

    if(isClockwise == True):
        if cross_product > 0:
            angle_deg = 360 - angle_deg
    elif (isClockwise == False):
        if cross_product < 0:
            angle_deg = 360 - angle_deg
    
    return angle_deg

