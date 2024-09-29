import os
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import csv
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExtractor:
    """
    A class to extract features from binary mask images and process them for machine learning.

    Attributes
    ----------
    image_directory : str
        Directory containing the binary mask images.
    labels_csv_path : str
        Path to the CSV file containing labels.
    output_path : str
        Path to save the processed data.
    num_features : int
        Number of top features to select.
    labels : dict
        Dictionary of labels with patient IDs as keys and labels as values.
    """

    def __init__(self, image_directory, labels_csv_path, output_path, num_features=10):
        """
        Initialize the FeatureExtractor with directories and paths.

        Parameters
        ----------
        image_directory : str
            Directory containing the binary mask images.
        labels_csv_path : str
            Path to the CSV file containing labels and codes.
        output_path : str
            Path to save the processed data.
        num_features : int, optional
            Number of top features to select (default is 10).
        """
        self.image_directory = image_directory
        self.labels_csv_path = labels_csv_path
        self.output_path = output_path
        self.num_features = num_features
        self.labels = self.read_labels()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Validate paths
        self.validate_paths()

    def validate_paths(self):
        """
        Validate the provided paths for image directory and labels CSV file.
        """
        if not os.path.isdir(self.image_directory):
            self.logger.error(f"Image directory {self.image_directory} does not exist.")
            raise FileNotFoundError(f"Image directory {self.image_directory} does not exist.")
        if not os.path.isfile(self.labels_csv_path):
            self.logger.error(f"Labels CSV file {self.labels_csv_path} does not exist.")
            raise FileNotFoundError(f"Labels CSV file {self.labels_csv_path} does not exist.")

    def read_labels(self):
        """
        Read the labels and codes from the CSV file.

        Returns
        -------
        dict
            Dictionary of labels with patient IDs as keys and a tuple (code, label) as values.
        """
        labels = {}
        try:
            with open(self.labels_csv_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                for row in reader:
                    id_image, code, label = row
                    labels[id_image] = (code, label)
        except Exception as e:
            self.logger.error(f"Error reading labels CSV file: {e}")
            raise
        return labels

    def calculate_features(self, image_path):
        """
        Calculate features from a binary mask image.

        Parameters
        ----------
        image_path : str
            Path to the binary mask image.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the features for the given image.
        """
        try:
            # Load the binary mask image
            binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                self.logger.warning(f"Image at path {image_path} could not be read.")
                return pd.DataFrame()

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return pd.DataFrame([[0] * len(self.get_feature_names())], columns=self.get_feature_names())

            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)
            features = self.calculate_contour_features(contour, binary_mask)

            # Return the features as a DataFrame
            return pd.DataFrame([features], columns=self.get_feature_names())

        except Exception as e:
            self.logger.error(f"Error calculating features for image {image_path}: {e}")
            return pd.DataFrame()

    def calculate_contour_features(self, contour, binary_mask):
        """
        Calculate contour-based features.

        Parameters
        ----------
        contour : array
            Contour from the binary mask image.
        binary_mask : ndarray
            Binary mask image.

        Returns
        -------
        list
            List of calculated features.
        """
        # Basic contour features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        extent = float(area) / (w * h) if w * h != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = 0, 0

        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        compactness = (perimeter ** 2) / area if area != 0 else 0

        # Fit an ellipse to the contour if possible
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x_, y_), (MA, ma), angle = ellipse
            ellipse_area = np.pi * MA * ma / 4
        else:
            MA = ma = angle = ellipse_area = 0

        # Calculate the mean height of the contour
        topmost = np.min(contour[:, 0, 1])
        bottommost = np.max(contour[:, 0, 1])
        mean_height = (topmost + bottommost) / 2

        # Calculate the x-coordinates of pixels along the mean height inside the contour
        width_img = binary_mask.shape[1]
        x_inside = [x for x in range(width_img) if cv2.pointPolygonTest(contour, (x, mean_height), False) >= 0]

        # Find the longest continuous strip of pixels at mean height
        if len(x_inside) > 0:
            strips = np.split(x_inside, np.where(np.diff(x_inside) != 1)[0] + 1)
            longest_strip = max(strips, key=len)
            pixels_at_height_mean = len(longest_strip)
        else:
            pixels_at_height_mean = 0

        # Compile all features into a list
        features = [
            area, perimeter, h, w, aspect_ratio, extent, solidity, compactness, cx, cy,
            MA, ma, angle, ellipse_area, pixels_at_height_mean, *hu_moments
        ]

        return features

    def get_feature_names(self):
        """
        Get the feature names for the features calculated.

        Returns
        -------
        list
            List of feature names.
        """
        feature_names = [
            'Area', 'Perimeter', 'Height', 'Width', 'Aspect_Ratio', 'Extent', 'Solidity', 'Compactness', 'Centroid_X',
            'Centroid_Y', 'Major_Axis', 'Minor_Axis', 'Orientation', 'Ellipse_Area', 'Pixels_at_height_mean',
            *[f'Hu_{i+1}' for i in range(7)]
        ]
        return feature_names

    def extract_features(self):
        """
        Extract features for all images in the directory.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the features and labels for all images.
        """
        data = []
        for filename in os.listdir(self.image_directory):
            if filename.endswith('.jpg'):
                # Extract ID from filename and calculate features
                id_image = filename.split('_')[-1].split('.')[0]
                image_path = os.path.join(self.image_directory, filename)
                features_df = self.calculate_features(image_path)
                if not features_df.empty:
                    code, label = self.labels.get(id_image, ('unknown', 'unknown'))
                    if label != 'unknown':  # Only include images with known labels
                        features_df['ID'] = id_image
                        features_df['Code'] = code
                        features_df['Label'] = label
                        data.append(features_df)

        if not data:
            self.logger.warning("No valid data to process.")
            return pd.DataFrame()

        return pd.concat(data, ignore_index=True)

    def save_processed_data(self, df):
        """
        Save the processed data to .joblib and .csv files, and return the selected features and their scores.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the original features and labels.

        Returns
        -------
        tuple
            Tuple containing the selected features, labels, scaler, selector, sorted features, and their scores.
        """
        if df.empty:
            self.logger.error("Dataframe is empty. Exiting without saving data.")
            return None

        df = df[df['Label'] != 'unknown']

        if df.empty:
            self.logger.error("No valid labels found. Exiting without saving data.")
            return None

        # Define the directory where files will be saved
        save_dir = os.path.join(os.path.dirname(__file__), 'processed_data')
        os.makedirs(save_dir, exist_ok=True)

        # Extract the feature matrix (X) and labels (y)
        X = df.drop(columns=['Label', 'ID', 'Code'], errors='ignore')
        y = df['Label'].astype(int)

        # Save the original values, including 'ID', 'Code', and 'Label' columns, to CSV
        original_csv_path = os.path.join(save_dir, 'original_features.csv')
        df[['ID', 'Code'] + X.columns.tolist() + ['Label']].to_csv(original_csv_path, index=False)
        self.logger.info(f"Original feature values with Code saved to {original_csv_path}")

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select top features using ANOVA F-test
        selector = SelectKBest(f_classif, k=self.num_features)
        X_selected = selector.fit_transform(X_scaled, y)

        # Sort features by their F-test scores
        scores = selector.scores_
        sorted_indices = np.argsort(scores)[::-1][:self.num_features]
        sorted_features = X.columns[sorted_indices]
        sorted_scores = scores[sorted_indices]

        # Log the selected features and their scores
        feature_score_pairs = list(zip(sorted_features, sorted_scores))
        self.logger.info("Selected features and their scores:")
        for feature, score in feature_score_pairs:
            self.logger.info(f"Feature: {feature}, Score: {score}")

        # Save the standardized values to .joblib
        joblib_path = os.path.join(save_dir, 'standardized_features.joblib')
        joblib.dump((X_selected, y, scaler, selector,sorted_features, sorted_scores ), joblib_path)
        self.logger.info(f"Standardized features and model saved to {joblib_path}")

        # Return the selected features, labels, scaler, selector, and their scores
        return X_selected, y, scaler, selector, sorted_features, sorted_scores


    def save_features_to_csv(self, csv_filename='processed_data.csv'):
        """
        Extract features for all images and save them to a CSV file.

        Parameters
        ----------
        csv_filename : str, optional
            Name of the CSV file to save the features (default is 'processed_data.csv').
        """
        # Extract features for all images
        df = self.extract_features()
        if df.empty:
            self.logger.warning("No features to save.")
            return pd.DataFrame()  # Return an empty DataFrame instead of None

        # Reorder columns to have 'ID', 'Code', features, and 'Label'
        columns = ['ID', 'Code'] + self.get_feature_names() + ['Label']
        result_df = df[columns]

        # Save the DataFrame to a CSV file
        output_csv_path = os.path.join('processed_data', csv_filename)
        os.makedirs('processed_data', exist_ok=True)
        result_df.to_csv(output_csv_path, index=False)

        self.logger.info(f"Features saved to {output_csv_path}")

        return result_df
    
    def run(self):
        """
        Run the feature extraction and processing pipeline.
        """
        df = self.extract_features()

        if df.empty:
            self.logger.error("No features extracted. Exiting.")
            return

        # Save the original and standardized data
        self.save_processed_data(df)
