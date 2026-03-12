# Deepfake Classification Project

## Overview
This project focuses on classifying deepfake media using a machine learning model. The objective is to analyze a dataset containing metadata about various media files and predict whether they are real or fake using a Random Forest Classifier.

## Dataset
The dataset `deepfake_detection_metadata_dataset.csv` contains 1000 rows of media metadata. It includes the following features:
- `media_type`: Image or Video
- `content_category`: News, Social Media, Interview, Political Speech
- `face_count`: Number of faces detected in the media
- `audio_present`: Whether audio is present
- `lip_sync_score`: Assessment of the lip sync quality
- `visual_artifacts_score`: Score indicating the presence of visual artifacts
- `compression_level`: Level of compression applied to the media
- `lighting_inconsistency_score`: Score evaluating lighting inconsistencies
- `source_platform`: Social media or news platform where the media was sourced
- `label`: Real or Fake

## Methodology
The analysis runs in a Jupyter Notebook (`deepfake_analysis.ipynb`) and covers the following steps:
1. **Data Loading and Exploration:** Loading the data using pandas.
2. **Data Cleaning:** Dropping irrelevant columns.
3. **Encoding:** Converting categorical data into numeric values (One-Hot Encoding) and mapping the target label (Real to 0, Fake to 1).
4. **Feature Scaling:** Preprocessing numerical features using `StandardScaler`.
5. **Model Training:** Splitting the data into training (80%) and testing (20%) sets, then training a `RandomForestClassifier`.
6. **Evaluation:** The model is evaluated on the test set. An initial baseline using all features performs exceptionally well, while a more realistic evaluation excluding direct predictive artifacts yields performance closer to baseline, demonstrating the challenges in deepfake detection.

## Requirements
- Python 3
- pandas
- scikit-learn
- Jupyter Notebook

## Running the Project
1. Activate the provided virtual environment (`venv`).
2. Ensure required packages are installed (e.g., `pip install pandas scikit-learn`).
3. Start a Jupyter server and open `deepfake_analysis.ipynb`.
4. Run the notebook cells sequentially to reproduce the workflow.
