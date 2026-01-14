import cv2
import numpy as np

def load_image(path, size=(128, 128)):
    """
    Load an image from path, resize it to a fixed size.
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img_resized = cv2.resize(img, size)
        return img_resized
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Compute the Color Histogram using HSV color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_edge_features(image):
    """
    Compute percentage of edge pixels using Canny Edge Detection.
    Returns a small feature vector (mean edge density, edge variance per quadrant).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Feature 1: Total Edge Density
    total_pixels = edges.size
    edge_pixels = np.count_nonzero(edges)
    density = edge_pixels / total_pixels
    
    # Feature 2-5: Split into 4 quadrants and get density for each
    h, w = edges.shape
    h2, w2 = h // 2, w // 2
    
    q1 = edges[0:h2, 0:w2]
    q2 = edges[0:h2, w2:w]
    q3 = edges[h2:h, 0:w2]
    q4 = edges[h2:h, w2:w]
    
    densities = [
        np.count_nonzero(q) / q.size for q in [q1, q2, q3, q4]
    ]
    
    return np.array([density] + densities)

def extract_hog_features(image):
    """
    Compute Histogram of Oriented Gradients (HOG) features.
    """
    from skimage.feature import hog
    
    # Resize to standard size specifically for HOG
    img_resized = cv2.resize(image, (64, 128)) 
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, transform_sqrt=True)
    return features
