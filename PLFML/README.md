# Diabetes Prediction Pipeline

This project showcases a **machine learning pipeline** for predicting diabetes progression using the **Diabetes dataset** from `sklearn.datasets`. The pipeline is designed to easily experiment with different configurations, such as feature selection, data scaling, and model choices, to optimize prediction accuracy. It includes components for data loading, preprocessing, model training, and performance evaluation, and allows for comparing results across different setups.

### 

### Features:

* **Flexible Experimentation**: Configure and test different setups by changing:

  * **Feature Selection**: Select specific features from the dataset to use in the model.
  * **Scaling Methods**: Try different scalers like **MinMax**, **Standard**, and **Robust** to preprocess the data.
  * **Regression Models**: Choose between **Linear Regression**, **Ridge**, **Lasso**, and **Random Forest Regressor**.

* **Model Evaluation**: Evaluate model performance using standard metrics:

  * **R²** (Coefficient of Determination)
  * **RMSE** (Root Mean Squared Error)
  * **MAE** (Mean Absolute Error)

* **Model Comparison**: Easily compare the results of different configurations to determine the best preprocessing and model combination.

* **Results Saving**: The trained model and evaluation metrics are saved for future use and easy access.

### Components

1. **DataLoaderComponent**: Loads the Diabetes dataset and allows for:

   * Shuffling the data.
   * Selecting specific features.
   * Taking a subset of the data.

2. **PreprocessorComponent**: Preprocesses the data by:

   * Scaling using **MinMax**, **Standard**, or **Robust** scalers.
   * Optionally applying a **log transformation** to the features.

3. **RegressorComponent**: Trains a regression model. Models available:

   * **Linear Regression**
   * **Ridge Regression**
   * **Lasso Regression**
   * **Random Forest Regressor**

4. **EvaluatorComponent**: Evaluates the trained model using performance metrics:

   * **R²**
   * **RMSE**
   * **MAE**

5. **SimpleRegressionWorkflow**: Orchestrates the pipeline by running the components in sequence:

   * Loads the data.
   * Applies preprocessing.
   * Trains the model.
   * Evaluates performance.
   * Saves the trained model and evaluation metrics.
