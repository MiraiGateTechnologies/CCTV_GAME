"""
Geometry helper functions for the CCTV Vehicle Counter.
Extracted to adhere to Single Responsibility Principle (SRP).
"""
import numpy as np
import cv2

def ccw(A, B, C):
    """Check if three points are listed in counter-clockwise order."""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def is_intersect(A, B, C, D):
    """Check if line segment AB intersects line segment CD."""
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_in_polygon(pt, polygon):
    """Check if a point is inside a polygon using OpenCV."""
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0

def polygon_bounding_box(polygon, frame_h, frame_w):
    """Calculate the bounding box of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (max(0, int(min(xs))), max(0, int(min(ys))),
            min(frame_w, int(max(xs))), min(frame_h, int(max(ys))))
