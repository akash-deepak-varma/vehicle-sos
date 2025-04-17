
import numpy as np
from sklearn.cluster import KMeans

# List of common color names and their RGB values
COLOR_NAMES = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (169, 169, 169),
    "brown": (165, 42, 42),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "purple": (128, 0, 128),
    "violet": (238, 130, 238),
    "indigo": (75, 0, 130),
}

# Function to calculate the Euclidean distance between two RGB colors
def euclidean_distance(rgb1, rgb2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

# Function to get the closest color name from the predefined list
def rgb_to_name(rgb):
    min_distance = float('inf')
    closest_color = None

    for color_name, color_value in COLOR_NAMES.items():
        distance = euclidean_distance(rgb, color_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

# Function to get the dominant color from an image using K-means clustering
def get_dominant_color(image, k=1):
    """
    Returns the dominant color in the image using k-means clustering.
    
    Parameters:
        image (numpy array): The input image (BGR).
        k (int): The number of clusters for k-means. Default is 1 for the most dominant color.
        
    Returns:
        tuple: The dominant color in RGB format.
    """
    # Reshaping the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the RGB color of the cluster center
    dominant_color = kmeans.cluster_centers_[0]
    
    # Convert BGR to RGB
    dominant_color = dominant_color[::-1]
    
    return tuple(map(int, dominant_color))