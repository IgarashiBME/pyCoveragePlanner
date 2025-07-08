import numpy as np

## parameter
rotate = 0.00
INTERPOLATE_LENGTH = 5.0 # meter
DIVZERO_PROTECT = 0.000000000001

def distance(p1,p2):
    return np.linalg.norm(np.array([p2[0], p2[1]])-np.array([p1[0], p1[1]]))

def transform(x, y, tx, ty, deg, sx, sy):
    deg = deg * np.pi /180.0
    # sx, sy are not used in the main logic, so their error handling is omitted for brevity.
    return [sx*((x-tx)*np.cos(deg) -(y-ty)*np.sin(deg)) +tx,\
            sy*((x-tx)*np.sin(deg) +(y-ty)*np.cos(deg)) +ty]

def calculate_min(pts):
    min_x = np.amin(pts, axis=0)[0]
    min_y = np.amin(pts, axis=0)[1]

    if min_x > 0:
        min_x = 0.0
    if min_y > 0:
        min_y = 0.0

    return min_x, min_y

def translation(pts):
    min_x, min_y = calculate_min(pts)
    trans_array = np.tile([min_x, min_y], (pts.shape[0], 1))
    trans_pts = pts - trans_array
    return min_x, min_y, trans_pts

def calcPointInLineWithY(p1, p2, y):
    # This function calculates the x-coordinate on the line segment p1-p2 for a given y.
    # Check if the horizontal line at y even intersects the segment's y-range.
    if not (min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])):
        return False
        
    # Handle vertical lines
    if abs(p1[0] - p2[0]) < DIVZERO_PROTECT:
        return [p1[0], y]
        
    # Handle horizontal lines
    if abs(p1[1] - p2[1]) < DIVZERO_PROTECT:
        return False # A horizontal line segment cannot intersect a different horizontal line.

    # Calculate intersection x
    s = p1[1] - p2[1]
    x = (y - p1[1]) * (p1[0] - p2[0]) / s + p1[0]

    # Check if the intersection point is within the x-bounds of the segment
    if not (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])):
        return False
        
    return [x, y]

def createPolygonBounds(points):
    if len(points) == 0:
        return {"center":(0,0), "nw":(0,0), "ne":(0,0), "sw":(0,0), "se":(0,0)}
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])

    center = ((max_x + min_x) / 2.0, (max_y + min_y) / 2.0)
    nw = (min_x, max_y)
    ne = (max_x, max_y)
    sw = (min_x, min_y)
    se = (max_x, min_y)
    return {"center": center, "nw": nw, "ne": ne, "sw": sw, "se": se}

## The original calcLatsInPolygon function was the source of the bug and is now removed.
# def calcLatsInPolygon(rect, spacing):

def si(i, j):
    if i > j - 1:
        return i - j
    if i < 0:
        return i + j
    return i

def createRotatePolygon(path, bounds, rotate):
    res = []
    center_x, center_y = bounds["center"]
    for i in range(len(path)):
        tr = transform(path[i][0], path[i][1], center_x, center_y, -rotate, 1.0, 1.0)
        res.append(tr)
    return np.array(res)

# ==============================================================================
# BUG FIXED AND REFACTORED VERSION of calculate()
# ==============================================================================
def calculate(pts, spacing=0.6):
    if pts.shape[0] < 3:
        # Not a polygon, cannot calculate path
        return np.array([]), np.array([])

    ## 1. Translate polygon so all coordinates are positive
    min_pts_x, min_pts_y, trans_pts = translation(pts)

    ## 2. Get the center of the UN-ROTATED polygon for consistent rotation
    unrotated_bounds = createPolygonBounds(trans_pts)
    rotation_center = unrotated_bounds["center"]

    ## 3. Rotate the polygon by the desired angle
    # Note: The original code used a negative rotate value here, and positive for un-rotating.
    # The transform function also uses a negative angle internally for standard rotation.
    # To avoid double negatives, we pass the desired rotation angle directly.
    # We will pass -rotate to un-rotate it later.
    rPolygon = createRotatePolygon(trans_pts, {"center": rotation_center}, rotate)

    ## 4. Create a bounding box for the ROTATED polygon. This is the key change.
    rotated_bounds = createPolygonBounds(rPolygon)

    ## 5. Generate scanlines based on the ROTATED bounding box and the correct `spacing`
    path_in_rotated_space = []
    y_max = rotated_bounds["ne"][1]  # Top of the rotated bounding box
    y_min = rotated_bounds["se"][1]  # Bottom of the rotated bounding box
    
    # Calculate the number of lines needed to cover the vertical span
    num_lines = int(np.floor((y_max - y_min) / spacing))

    ## 6. Traverse each scanline and find intersections with the rotated polygon
    for i in range(num_lines + 2):  # Add a small buffer to ensure covering edges
        # The Y-coordinate for the current horizontal scanline. The step is now `spacing`.
        scan_y = y_max - (i * spacing)
        
        intersections = []
        for j in range(len(rPolygon)):
            p1 = rPolygon[j]
            p2 = rPolygon[si(j + 1, len(rPolygon))]
            
            # Find an intersection point
            point = calcPointInLineWithY(p1, p2, scan_y)
            if point:
                intersections.append(point)

        if len(intersections) < 2:
            continue

        # Sort intersections by x-coordinate to get the start and end of the line
        intersections.sort(key=lambda p: p[0])
        start_point = intersections[0]
        end_point = intersections[-1] # Use the last point in case of concavity

        # Create zig-zag (boustrophedon) path
        if i % 2: # Odd lines (e.g., 1, 3, 5...)
            path_in_rotated_space.append(end_point)
            path_in_rotated_space.append(start_point)
        else: # Even lines (e.g., 0, 2, 4...)
            path_in_rotated_space.append(start_point)
            path_in_rotated_space.append(end_point)

    if not path_in_rotated_space:
        return np.array([]), np.array([])
        
    path_in_rotated_space = np.array(path_in_rotated_space)
    
    ## 7. Rotate the generated path back to the original orientation
    path_in_translated_space = createRotatePolygon(path_in_rotated_space, {"center": rotation_center}, -rotate)
    
    ## 8. Translate the path back to its original coordinate system
    trans_array = np.tile([min_pts_x, min_pts_y], (path_in_translated_space.shape[0], 1))
    final_path = path_in_translated_space + trans_array

    return final_path, path_in_translated_space


def interpolate(path):
    """
    Corrected interpolate function. It now works on the final path.
    """
    if path.shape[0] < 2:
        return path

    interpolated_path = []
    # Add the first point
    interpolated_path.append(path[0])

    for i in range(path.shape[0] - 1):
        p1 = path[i]
        p2 = path[i+1]
        dist = distance(p1, p2)
        
        # Avoid division by zero and unnecessary calculations
        if dist < INTERPOLATE_LENGTH:
            interpolated_path.append(p2)
            continue
            
        # Calculate number of points needed for this segment
        num_points = int(dist / INTERPOLATE_LENGTH)
        
        # np.linspace includes start and end. We add points from after p1 up to p2.
        x_latent = np.linspace(p1[0], p2[0], num_points + 1)[1:]
        y_latent = np.linspace(p1[1], p2[1], num_points + 1)[1:]
        
        interpolated_path.extend(np.column_stack([x_latent, y_latent]))

    return np.array(interpolated_path)


## Example Usage:
"""if __name__ == '__main__':
    # Define a sample polygon (e.g., a tilted rectangle)
    # This polygon is at a "specific x,y coordinate" away from the origin
    # to test the bug fix.
    sample_polygon = np.array([
        [50, 50],
        [100, 70],
        [110, 120],
        [60, 100]
    ])

    # 1. Calculate the main path vertices
    final_path_vertices, _ = calculate(sample_polygon)

    if final_path_vertices.size > 0:
        # 2. Interpolate the final path to get a dense path
        dense_path = interpolate(final_path_vertices)
    
        print("Calculation successful.")
        print(f"Generated {len(final_path_vertices)} vertices.")
        print(f"Interpolated to {len(dense_path)} points.")

        # You can save the output to a CSV to visualize and verify
        # np.savetxt('final_path.csv', final_path_vertices, fmt='%.7f', delimiter=',')
        # np.savetxt('interpolated_path.csv', dense_path, fmt='%.7f', delimiter=',')
    else:
        print("Could not generate a path for the given polygon.")"""