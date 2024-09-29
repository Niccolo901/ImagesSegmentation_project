import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import logging
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

class CNNModel:
    """
    A class to develop, train, and evaluate a CNN model for knee distension classification.
    
    Attributes
    ----------
    annotations_path : str
        Path to the annotations CSV file.
    image_directory : str
        Directory containing the binary mask images.
    output_directory : str
        Directory to save the trained model and results.
    model_type : str
        Type of model to use ('custom' for the custom CNN or 'resnet50' for ResNet50).
    """
    
    def __init__(self, annotations_path, image_directory, output_directory, model_type='custom'):
        """
        Initialize the CNNModel with paths and load image data.
        
        Parameters
        ----------
        annotations_path : str
            Path to the annotations CSV file.
        image_directory : str
            Directory containing the binary mask images.
        output_directory : str
            Directory to save the trained model and results.
        model_type : str
            Type of model to use ('custom' for the custom CNN or 'resnet50' for ResNet50).
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.annotations_path = annotations_path
        self.image_directory = image_directory
        self.output_directory = output_directory
        self.model_type = model_type
        self.images, self.labels, self.groups = self.load_images_and_labels()

    def load_images_and_labels(self):
        """
        Load images and corresponding labels from the specified directory and CSV file.
        
        Returns
        -------
        tuple
            Tuple containing the images, labels, and groups (patient codes).
        """
        self.logger.info("Loading images and labels...")
        
        annotations = pd.read_csv(self.annotations_path)
        images = []
        labels = []
        groups = []
        
        for idx, row in annotations.iterrows():
            image_id = row['ID']
            label = row['First Character']
            patient_code = row['Code']
            
            # Search for the image file that contains the ID
            image_file = None
            for file_name in os.listdir(self.image_directory):
                if f"_{image_id}.jpg" in file_name:
                    image_file = file_name
                    break
            
            if image_file:
                image_path = os.path.join(self.image_directory, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128, 128))  # Resize to a fixed size
                if self.model_type == 'resnet50':
                    # Convert grayscale to 3 channels for ResNet50
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    image = preprocess_input(image)  # Preprocess for ResNet50
                images.append(image)
                labels.append(label)
                groups.append(patient_code)
            else:
                self.logger.warning(f"Image with ID {image_id} not found.")
        
        images = np.array(images)
        if self.model_type == 'custom':
            images = np.expand_dims(images, axis=-1)  # Add channel dimension for custom CNN
        
        labels = np.array(labels)
        groups = np.array(groups)
        
        return images, labels, groups
    
    def build_model(self, input_shape, num_classes):
        """
        Build a CNN model or ResNet50 model based on the specified type.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of the input images.
        num_classes : int
            Number of classes.
        
        Returns
        -------
        model
            Compiled CNN or ResNet50 model.
        """
        if self.model_type == 'custom':
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
        elif self.model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
            base_model.trainable = False  # Freeze the base model layers
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def lr_schedule(self, epoch, lr):
        """
        Learning rate scheduler function.
        
        Parameters
        ----------
        epoch : int
            The current epoch number.
        lr : float
            The current learning rate.
        
        Returns
        -------
        float
            Updated learning rate.
        """
        if epoch > 10:
            lr = lr * 0.5  # Reduce learning rate after 10 epochs
        return lr
    
    def train_and_evaluate_model(self):
        """
        Train and evaluate the model using GroupKFold cross-validation.

        Returns
        -------
        best_model : model
            The best model based on validation accuracy.
        best_accuracy : float
            The best accuracy achieved during cross-validation.
        best_fold_no : int
            The fold number of the best model.
        """
        self.logger.info("Preparing data for GroupKFold cross-validation...")

        labels = to_categorical(self.labels)
        group_kfold = GroupKFold(n_splits=5)

        fold_no = 1
        best_model = None
        best_accuracy = 0
        best_fold_no = 0
        all_y_true = []
        all_y_pred = []

        for train_index, test_index in group_kfold.split(self.images, self.labels, self.groups):
            self.logger.info(f"Training on fold {fold_no}...")
            X_train, X_test = self.images[train_index], self.images[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model = self.build_model(input_shape=(128, 128, 3) if self.model_type == 'resnet50' else (128, 128, 1), 
                                     num_classes=len(np.unique(self.labels)))

            self.logger.info(f"Training model for fold {fold_no}...")

            # Callbacks: Learning Rate Scheduler and Early Stopping
            lr_scheduler = LearningRateScheduler(self.lr_schedule)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Training the model
            history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), 
                                callbacks=[lr_scheduler, early_stopping], verbose=1)

            self.logger.info(f"Evaluating model for fold {fold_no}...")
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred_classes)

            # Evaluate the model
            accuracy = np.mean(y_pred_classes == y_true)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_fold_no = fold_no

            fold_no += 1

        self.logger.info("Generating combined classification report...")
        self.logger.info("Classification report:\n%s", classification_report(all_y_true, all_y_pred))

        self.logger.info("Generating combined confusion matrix...")
        conf_matrix = confusion_matrix(all_y_true, all_y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Save the best model only
        best_model_save_path = os.path.join(self.output_directory, f'model_fold_{best_fold_no}_{self.model_type}.h5')
        self.logger.info(f"Saving the best model (Fold {best_fold_no}) to {best_model_save_path}...")
        best_model.save(best_model_save_path)

        self.logger.info(f"Best model saved with accuracy: {best_accuracy:.4f}")
        return best_model, best_accuracy, best_fold_no

    
    def run_training_pipeline(self):
        """
        Run the entire CNN training and evaluation pipeline with GroupKFold cross-validation.
        """
        best_model, best_accuracy, best_fold_no = self.train_and_evaluate_model()
        # The best model is already saved within the train_and_evaluate_model function.
        self.logger.info(f"Best model from fold {best_fold_no} saved with accuracy: {best_accuracy:.4f}")
