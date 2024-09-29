# Comparative Analysis of Manual vs. Automatic Feature Extraction in Ultrasound Musculoskeletal Segmentation Masks

## Project Overview

This project, authored by Niccolò Cibei, focuses on classifying knee recess segmentation masks from ultrasound images into two categories: **distended** or **not-distended**. The goal is to analyze and compare the effectiveness of manual vs. automatic feature extraction techniques for this classification task.

### Objectives

1. **Manual Feature Extraction + SVM Classification**  
   Features such as area, perimeter, and shape descriptors are manually extracted from binary masks and used to train a Support Vector Machine (SVM) for classification.

2. **Automatic Feature Extraction using CNN**  
   Convolutional Neural Networks (CNNs) are employed to automatically extract features and classify the masks without manual intervention.

## Dataset

- **Composition**: 916 binary masks representing knee recesses from ultrasound images of hemophilic patients.
- **Annotations**: Binary labels (1 for "distended" and 0 for "not-distended") are provided in an `annotations.csv` file.

## Methodology

### 1. Manual Feature Extraction + SVM Classification

- **Feature Extraction**: Key features such as area, perimeter, compactness, and Hu moments are manually extracted.
- **Model Training**: An SVM model is trained on the manually extracted features, with hyperparameters optimized via GridSearchCV.
- **Evaluation**: The model achieves an overall accuracy of 88%, performing well on non-distended cases but struggling with distended recesses.

### 2. CNN Model Development

- **Models**: Both a pre-trained ResNet50 and a custom CNN are developed and evaluated.
- **Training**: The CNN models are trained using GroupKFold cross-validation to prevent patient data leakage.
- **Evaluation**: The ResNet50 achieves an accuracy of 90%, outperforming the custom CNN. However, both models find classifying distended recesses challenging.

## Results

- **SVM Model**: Accuracy = 88%, with strong performance on non-distended cases.
- **ResNet50**: Accuracy = 90%, leveraging pre-trained features for superior performance.
- **Custom CNN**: Accuracy = 87%, showing potential but trailing behind ResNet50.

## Conclusion

While both manual and automatic feature extraction methods demonstrate promising results, the ResNet50 CNN model outperforms others. Future work should focus on improving the classification of distended recesses, possibly by refining feature extraction techniques or enhancing the CNN architectures.

---

### File Structure
- **Dataset**: Contains binary mask images and `annotations.csv`.
- **Manual_Feature_Extraction_SVM.ipynb**: Notebook for manual feature extraction and SVM model training.
- **CNN_Model_Development.ipynb**: Notebook for CNN model training and evaluation.
