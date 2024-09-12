**Image Classification using Support Vector Machine (SVM)**

This repository contains code for classifying images into two categories: `empty` and `not_empty`. It uses a Support Vector Machine (SVM) with a grid search to find the best hyperparameters for the model. The main features of this project include loading image data, preprocessing, splitting data into training and test sets, and training the model using Scikit-learn.

How it Works:
1. Data Loading:
   - The images are loaded from the specified directory (`clf-data`), which contains subfolders named `empty` and `not_empty`.
   - Only valid image file formats (e.g., `.jpg`, `.png`, etc.) are processed.

2. Preprocessing:
   - Each image is resized to 15x15 pixels to reduce dimensionality.
   - The images are then flattened into 1D arrays for model training.

3. Train/Test Split:
   - The data is split into training and testing sets (80/20), using stratified sampling to maintain label distribution.

4. Model Training:
   - A Support Vector Machine (SVC) model is trained using GridSearchCV to tune the hyperparameters (`gamma` and `C`).
   - Cross-validation (5-fold) is performed to find the optimal parameters.

5. Model Evaluation:
   - The trained model is evaluated on the test set, and the accuracy score is printed.

6. Saving the Model:
   - The trained and optimized model is saved using `pickle` for future use.

Dependencies:
- Python 3
- Scikit-learn
- Scikit-image
- NumPy

Usage:
1. Clone the repository and navigate to the project directory.
2. Place your image data in the `clf-data` folder, with subdirectories for each category (`empty`, `not_empty`).
3. Run the script to train the classifier.
4. The trained model will be saved as `modelIC.p`.

