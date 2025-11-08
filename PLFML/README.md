# 🖥️ Project: Diabetes Progression Prediction

### 📦 Setup Guide

**(It’s recommended to create a new Python environment for the setup, but not strictly necessary)**

1. **Download the project folder**:

   * Download the folder `PLF_SetUps/PLFML/`.

2. **Keep the directory structure intact**:

   * Ensure that the folder structure remains as is for everything to work properly.

3. **Run the Setup**:

   * Open and follow the instructions in the `setup.ipynb` file to set up the pipeline and dependencies.

4. **Install Required Libraries**:

   * Install all required libraries as mentioned in the following files:

     * `MLComps/myComps.py`

   To install the required dependencies, run:

   ```bash
   pip install PyLabFlow scikit-learn numpy joblib
   ```

5. **Debug/Check if all components are working**:

   * Open and run `deBug.ipynb` to verify that all components are functioning correctly.

6. **Experiment and Compare**:

   * Open and run `exptrail.ipynb`, modifying the configuration and `pplid` values.
   * Ensure that the components defined inside `MLComps/myComps.py` support the configuration changes.

---

### 🚀 Project Overview

This project is a **Diabetes Progression Prediction Pipeline** built using a flexible, modular structure. The pipeline includes components for:

* **Loading the dataset**: Fetching the Diabetes dataset and managing data splits.
* **Preprocessing the data**: Various data preprocessing techniques, including feature scaling and transformation.
* **Training different regression models**: Train multiple regression models such as **Linear Regression**, **Ridge Regression**, **Lasso Regression**, and **Random Forest**.
* **Evaluating model performance**: Evaluate the models using metrics like **R²**, **RMSE**, and **MAE**.

You can experiment with different configurations and models. For example:

* **Feature Selection**: Choose which features to include.
* **Scaling Methods**: Experiment with **MinMax**, **Standard**, or **Robust** scaling.
* **Models**: Compare different models, including **Linear Regression**, **Ridge**, **Lasso**, and **Random Forest**.

The pipeline allows you to easily **compare results** across different configurations and **save the trained models** and **evaluation metrics** for later use.

---

### 🧑‍🔬 Experimentation

Feel free to experiment with different configurations and explore how changing parameters affects model performance:

* **Try different scalers**:

  * Example: `MinMaxScaler` vs. `RobustScaler`.
  * See how each scaling technique affects model accuracy.
* **Test various regression models**:

  * Example: `RandomForestRegressor` vs. `LinearRegression`.
  * Compare performance across multiple models.
* **Compare different evaluation metrics**:

  * Example: `R²`, `RMSE`, and `MAE`.
  * Experiment with different evaluation criteria to assess the best-performing model.

---

This flexible setup allows you to easily experiment with various configurations and models, providing an opportunity to fine-tune the diabetes progression prediction for optimal results.

