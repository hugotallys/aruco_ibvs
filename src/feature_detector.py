import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, image_resolution=512):
        self.image_resolution = image_resolution
        
    def detect_markers_by_color(self, image):
        """
        Detects the centroids of red, green, blue, and magenta markers in the image.
        
        :param image: RGB image as a numpy array (shape: [height, width, 3]).
        :return: Dictionary with keys 'red', 'green', 'blue', 'magenta' and values as [x,y] centroids
                or None if the marker is not detected.
        """
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Red is tricky in HSV because it wraps around 0/180
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Green HSV range
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        # Blue HSV range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        
        # Magenta HSV range
        lower_magenta = np.array([140, 100, 100])
        upper_magenta = np.array([170, 255, 255])
        
        results = {}
        
        # Detect red marker (need to handle two ranges due to HSV wrap-around)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 | red_mask2  # Combine the two masks
        results['red'] = self._find_centroid(red_mask)
        
        # Detect green marker
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        results['green'] = self._find_centroid(green_mask)
        
        # Detect blue marker
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        results['blue'] = self._find_centroid(blue_mask)
        
        # Detect magenta marker
        magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
        results['magenta'] = self._find_centroid(magenta_mask)
        
        return results

    def _find_centroid(self, mask):
        """
        Find the centroid of the largest blob in the mask.
        
        :param mask: Binary mask image
        :return: [x, y] centroid coordinates or None if no blob found
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid only if the contour area is significant
        if cv2.contourArea(largest_contour) > 20:  # Minimum area threshold
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:  # Avoid division by zero
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return [cx, cy]
        
        return None
    
    def rotate_features(self, p, theta):
        """
        Rotate feature points around the image center.
        
        :param p: numpy array of points with shape [n, 2]
        :param theta: rotation angle in radians
        :return: rotated points
        """
        p = p - (self.image_resolution / 2)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        p = (R @ (p.T)).T

        p = p + (self.image_resolution / 2)

        return p

    def translate_features(self, p, translation):
        """
        Translate feature points.
        
        :param p: numpy array of points with shape [n, 2]
        :param translation: translation vector [tx, ty]
        :return: translated points
        """
        return p + translation

    def scale_features(self, p, scale):
        """
        Scale feature points relative to the image center.
        
        :param p: numpy array of points with shape [n, 2]
        :param scale: scale factor
        :return: scaled points
        """
        p = p - (self.image_resolution / 2)
        p = scale * p
        p = p + (self.image_resolution / 2)
        return p
    
    def get_detected_features_as_array(self, detected, order=['red', 'green', 'blue', 'magenta']):
        """
        Convert detected features dictionary to a numpy array.
        
        :param detected: Dictionary with color keys and [x,y] values
        :param order: List specifying the order of features in the output array
        :return: numpy array of shape [n, 2] with feature coordinates
        """
        return np.array([detected[color] for color in order])
    
    def draw_features(self, image, detected_points, target_points=None, colors=None, radius=4):
        """
        Draw detected and optionally target feature points on the image.
        
        Parameters:
        - image: The image to draw on (will be modified in-place)
        - detected_points: Array of detected feature points with shape [n, 2]
        - target_points: Optional array of target feature points with shape [n, 2]
        - colors: List of BGR colors for each feature point. Defaults to [(0,255,0), (255,0,0), (0,0,255), (0,0,0)]
        - radius: Radius of the circles to draw
        
        Returns:
        - Modified image with features drawn
        """
        # Default colors if not provided (Green, Blue, Red, Black)
        if colors is None:
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
        
        # Ensure we have enough colors for all points
        if len(colors) < detected_points.shape[0]:
            colors = colors * (detected_points.shape[0] // len(colors) + 1)
        
        # Draw detected points
        for i, point in enumerate(detected_points):
            cv2.circle(image, tuple(point.astype(np.int32)), radius, colors[i], -1)
        
        # Draw target points if provided
        if target_points is not None:
            for i, point in enumerate(target_points):
                cv2.circle(image, tuple(point.astype(np.int32)), radius, colors[i], -1)
        
        return image