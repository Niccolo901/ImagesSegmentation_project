import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import cv2

class ModelTrainer:
    def __init__(self, annotations_path, processed_data_path, output_directory, image_directory, original_images_directories):
        """
        Initialize the ModelTrainer class with paths and directories.

        Parameters
        ----------
        annotations_path : str
            Path to the CSV file containing annotations.
        processed_data_path : str
            Path to the file containing processed data (features and labels).
        output_directory : str
            Directory where output files (model, results) will be saved.
        image_directory : str
            Directory containing mask images.
        original_images_directories : list of str
            List of directories containing original images.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize all paths and directories
        self.annotations_path = annotations_path
        self.processed_data_path = processed_data_path
        self.output_directory = output_directory
        self.image_directory = image_directory
        self.original_images_directories = original_images_directories
        self.images_directory = os.path.join(output_directory, 'images')  # Set up the new images directory
        os.makedirs(self.images_directory, exist_ok=True)
        
        # Load data
        self.X_selected, self.y, self.selected_features = self.load_processed_data()
        self.groups = self.load_and_check_annotations()
        self.annotations = pd.read_csv(self.annotations_path)
        self.original_features_df = self.load_original_features()
        self.logger.info(f"Annotations columns: {self.annotations.columns.tolist()}")

    def load_processed_data(self):
        """
        Load the processed data from a joblib file.

        Returns
        -------
        X_selected : ndarray
            Selected features for training.
        y : ndarray
            Target labels.
        sorted_features : list
            List of selected feature names.
        """
        self.logger.info("Loading processed data...")
        X_selected, y, _, _, sorted_features, _ = joblib.load(self.processed_data_path)
        return X_selected, y, sorted_features

    def load_and_check_annotations(self):
        """
        Load and check the annotations for duplicates.

        Returns
        -------
        groups : ndarray
            Array of patient codes (used for grouping in GroupKFold).
        """
        annotations = pd.read_csv(self.annotations_path)
        self.logger.info("Annotations DataFrame:\n%s", annotations.head())
        
        # Check for duplicate patient codes
        duplicates = annotations[annotations.duplicated(subset='Code', keep=False)]
        if not duplicates.empty:
            self.logger.warning("Duplicate patient Codes found:\n%s", duplicates)
        
        # Extract groups 
        groups = annotations['Code'].values
        self.logger.info("Groups array:\n%s", groups[:10])
        return groups

    def load_original_features(self):
        """
        Load the original features from a CSV file.

        Returns
        -------
        original_features_df : DataFrame
            DataFrame containing original feature values.
        """
        self.logger.info("Loading original features from CSV...")
        csv_path = os.path.join(os.path.dirname(self.processed_data_path), 'original_features.csv')
        original_features_df = pd.read_csv(csv_path)
        return original_features_df

    def compute_class_weights(self):
        """
        Compute class weights to handle class imbalance.

        Returns
        -------
        dict
            Dictionary mapping class indices to their respective weights.
        """
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        return dict(enumerate(class_weights))

    def train_model(self):
        """
        Train the SVM model using GroupKFold cross-validation and GridSearchCV.

        Returns
        -------
        clf : SVC
            Trained SVM classifier with the best-found parameters.
        """
        self.logger.info("Computing class weights...")
        class_weights = self.compute_class_weights()

        self.logger.info("Setting up hyperparameter grid for GridSearchCV...")
        param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
        group_kfold = GroupKFold(n_splits=5)
        
        # Use GroupKFold with GridSearchCV to find the best hyperparameters
        grid = GridSearchCV(SVC(class_weight=class_weights), param_grid, 
                            refit=True, verbose=3, cv=group_kfold, scoring='f1', n_jobs=-1)

        self.logger.info("Fitting the model...")
        grid.fit(self.X_selected, self.y, groups=self.groups)

        # Extract the best model
        clf = grid.best_estimator_
        self.logger.info(f"Best kernel: {clf.kernel}")
        self.logger.info(f"Best parameters: {grid.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid.best_score_}")

        return clf

    def evaluate_model(self, clf):
        """
        Evaluate the trained model and save results including IDs and Codes.

        Parameters
        ----------
        clf : SVC
            Trained SVM classifier.

        Returns
        -------
        y_pred : ndarray
            Predicted labels by the classifier.
        all_indices : ndarray
            Indices corresponding to the original dataset.
        """
        self.logger.info("Evaluating the model on the entire dataset...")

        # Predict on the entire dataset after training
        y_pred = clf.predict(self.X_selected)
        all_indices = np.arange(len(self.y))

        self.logger.info("Classification report:\n%s", classification_report(self.y, y_pred))

        # Generate confusion matrix
        self.logger.info("Generating confusion matrix...")
        conf_matrix = confusion_matrix(self.y, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Calculate and plot ROC curve and AUC
        self.logger.info("Calculating ROC curve and AUC...")
        y_test_bin = pd.get_dummies(self.y).values
        y_pred_prob = clf.decision_function(self.X_selected)
        fpr, tpr, _ = roc_curve(y_test_bin[:, 1], y_pred_prob)
        roc_auc = auc(fpr, tpr)

        self.logger.info(f"ROC AUC: {roc_auc}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        # Save misclassified images
        self.save_misclassified_images(y_pred, all_indices)

        return y_pred, all_indices

    def save_misclassified_images(self, y_pred, all_indices):
        """
        Identify and save misclassified images based on the predictions.

        Parameters
        ----------
        y_pred : ndarray
            Predicted labels by the classifier.
        all_indices : ndarray
            Indices corresponding to the original dataset.
        """
        misclassified_indices_0 = all_indices[(self.y == 1) & (y_pred == 0)]
        misclassified_indices_1 = all_indices[(self.y == 0) & (y_pred == 1)]

        # Update paths to save misclassified images in the correct folders
        self.save_images(misclassified_indices_0, true_label=1, predicted_label=0)
        self.save_images(misclassified_indices_1, true_label=0, predicted_label=1)

    def save_images(self, indices, true_label, predicted_label):
        """
        Save images along with their true and predicted labels, and overlay feature values.

        Parameters
        ----------
        indices : list or ndarray
            List of indices of the images.
        true_label : int
            The true label of the image.
        predicted_label : int
            The predicted label of the image.
        """
        save_dir = os.path.join(self.images_directory, 'misclassified_images', f'labels_{true_label}')
        os.makedirs(save_dir, exist_ok=True)

        for idx in indices:
            id_image = self.annotations.iloc[idx]['ID']
            code = self.annotations.iloc[idx]['Code']
            img_filename = next((f for f in os.listdir(self.image_directory) if f.endswith(f"{id_image}.jpg")), None)
            if img_filename:
                img_path = os.path.join(self.image_directory, img_filename)
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                original_img_path = self.find_original_image_path(id_image, code)
                if original_img_path:
                    original_img = cv2.imread(original_img_path)
                    combined_img = self.combine_images_side_by_side(original_img, mask)

                    original_features = self.original_features_df[self.original_features_df['ID'] == id_image][self.selected_features].values.flatten()

                    combined_img = self.overlay_feature_values(combined_img, original_features, true_label, predicted_label)

                    save_path = os.path.join(save_dir, f"{id_image}_{code}_true{true_label}_pred{predicted_label}.jpg")
                    cv2.imwrite(save_path, combined_img)
                else:
                    self.logger.warning(f"Original image for ID {id_image} with code {code} not found in any fold directories.")
            else:
                self.logger.warning(f"Mask image for ID {id_image} not found")

    def find_original_image_path(self, id_image, code):
        """
        Find the path to the original image based on the image ID and code.

        Parameters
        ----------
        id_image : str
            Image ID to find the original image.
        code : str
            Code associated with the image.

        Returns
        -------
        str or None
            Path to the original image if found, else None.
        """
        for fold_dir in self.original_images_directories:
            original_img_path = os.path.join(fold_dir, f"{id_image}_{code}.jpg")
            if os.path.exists(original_img_path):
                return original_img_path
        return None

    def combine_images_side_by_side(self, original_img, mask):
        """
        Combine the original image and the mask side by side.

        Parameters
        ----------
        original_img : ndarray
            The original image.
        mask : ndarray
            The mask image (grayscale).

        Returns
        -------
        ndarray
            Combined image with the original image on the left and the mask on the right.
        """
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_img = np.concatenate((original_img, mask_bgr), axis=1)
        return combined_img

    def overlay_feature_values(self, img, feature_values, true_label, predicted_label):
        """
        Overlay the original feature values, true label, and predicted label onto the image.

        Parameters
        ----------
        img : ndarray
            Image on which to overlay the text.
        feature_values : list or ndarray
            The feature values to overlay on the image.
        true_label : int
            The true label of the image.
        predicted_label : int
            The predicted label of the image.

        Returns
        -------
        ndarray
            Image with the feature values, true label, and predicted label overlaid as text.
        """
        h, w = img.shape[:2]
        dy = 30

        # Overlay true and predicted labels
        text_label = f"True: {true_label}, Pred: {predicted_label}"
        text_size_label = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x_label = w - text_size_label[0] - 10
        y_label = h - 10
        cv2.putText(img, text_label, (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Overlay feature values on the right side of the image
        for i, value in enumerate(reversed(feature_values)):
            feature_name = self.selected_features[len(feature_values) - 1 - i]
            text = f"{feature_name}: {value:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x = w - text_size[0] - 10
            y = y_label - (i + 1) * dy
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return img

    def save_images_based_on_feature(self, feature_name, intersection_point):
        """
        Save images into folders based on the given feature's value relative to the intersection point.

        Parameters
        ----------
        feature_name : str
            The name of the feature to use for comparison with the intersection point.
        intersection_point : float
            The intersection point that separates the two classes.
        """
        # Directory structure for intersection-based images
        dir_0 = os.path.join(self.images_directory, f'{feature_name}_intersection', 'label_0_above_intersection')
        dir_1 = os.path.join(self.images_directory, f'{feature_name}_intersection', 'label_1_below_intersection')
        os.makedirs(dir_0, exist_ok=True)
        os.makedirs(dir_1, exist_ok=True)

        for idx in range(len(self.y)):
            id_image = self.annotations.iloc[idx]['ID']
            code = self.annotations.iloc[idx]['Code']
            feature_value = self.original_features_df[self.original_features_df['ID'] == id_image][feature_name].values[0]

            if self.y[idx] == 0 and feature_value > intersection_point:
                save_dir = dir_0
            elif self.y[idx] == 1 and feature_value < intersection_point:
                save_dir = dir_1
            else:
                continue

            img_filename = next((f for f in os.listdir(self.image_directory) if f.endswith(f"{id_image}.jpg")), None)
            if img_filename:
                img_path = os.path.join(self.image_directory, img_filename)
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                original_img_path = self.find_original_image_path(id_image, code)
                if original_img_path:
                    original_img = cv2.imread(original_img_path)
                    combined_img = self.combine_images_side_by_side(original_img, mask)

                    original_features = self.original_features_df[self.original_features_df['ID'] == id_image][self.selected_features].values.flatten()

                    combined_img = self.overlay_feature_values(combined_img, original_features, self.y[idx], self.y[idx])

                    save_path = os.path.join(save_dir, f"{id_image}_{code}_true{self.y[idx]}.jpg")
                    cv2.imwrite(save_path, combined_img)
                else:
                    self.logger.warning(f"Original image for ID {id_image} with code {code} not found in any fold directories.")
            else:
                self.logger.warning(f"Mask image for ID {id_image} not found")
        self.logger.info("Images saved")

    def save_model(self, clf):
        """
        Save the trained model to a file.

        Parameters
        ----------
        clf : SVC
            Trained SVM classifier.
        """
        self.logger.info("Saving the trained model...")
        joblib.dump(clf, os.path.join(self.output_directory, 'knee_recess_svm_classifier.joblib'))

    def save_results(self, clf, y_pred, all_indices):
        """
        Save the prediction results to a CSV file, including the ID and Code.

        Parameters
        ----------
        clf : SVC
            Trained SVM classifier.
        y_pred : ndarray
            Predicted labels by the classifier.
        all_indices : ndarray
            Indices corresponding to the original dataset.
        """
        self.logger.info("Saving the results to a CSV file...")
        results_df = pd.DataFrame(self.X_selected[all_indices], columns=[f'Feature_{i}' for i in range(self.X_selected.shape[1])])
        results_df['Label'] = self.y[all_indices]
        results_df['Predicted'] = y_pred[all_indices]
        results_df['ID'] = self.annotations.iloc[all_indices]['ID'].values
        results_df['Code'] = self.annotations.iloc[all_indices]['Code'].values
        results_df.to_csv(os.path.join(self.output_directory, 'results_with_predictions.csv'), index=False)

    def run(self):
        """
        Execute the full pipeline: train the model, evaluate it, and save the results.
        """
        clf = self.train_model()
        y_pred, all_indices = self.evaluate_model(clf)
        self.save_model(clf)
        self.save_results(clf, y_pred, all_indices)
