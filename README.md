# Customer Churn Prediction using Deep Learning

A machine learning project that predicts customer churn using Artificial Neural Networks (ANN). The project includes model training, hyperparameter tuning, and prediction capabilities for both classification and regression tasks.

## 🚀 Features

- **Binary Classification**: Predict customer churn (Exited: Yes/No)
- **Regression**: Predict estimated salary
- **Hyperparameter Tuning**: Automated grid search with cross-validation
- **Model Inference**: Ready-to-use prediction pipeline
- **Data Preprocessing**: Automated encoding and scaling

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.20.0
- scikit-learn
- pandas
- numpy

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DeepLearning_ANN
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate):
   ```python
   # Run experiments.ipynb
   # Trains ANN with architecture: 64 → 32 → 1 neurons
   ```

2. **Hyperparameter Tuning**:
   ```python
   # Run hyperparametertuningann.ipynb
   # Finds optimal neurons, layers, and epochs
   ```

3. **Regression Model** (Salary Prediction):
   ```python
   # Run salaryregression.ipynb
   # Trains regression ANN
   ```

### Making Predictions

```python
# Run prediction.ipynb
# Loads trained model and makes predictions on new data
```

### Web Application

```bash
streamlit run app.py
```

## 🏗️ Model Architecture

**Classification Model:**
- Input Layer: 12 features
- Hidden Layer 1: 64 neurons (ReLU)
- Hidden Layer 2: 32 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy

## 📊 Dataset

- **Source:** Churn_Modelling.csv
- **Samples:** 10,000 customer records
- **Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, etc.
- **Target:** Exited (Binary: 0/1)

## 🛠️ Technologies

- **Deep Learning:** TensorFlow/Keras
- **Machine Learning:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** TensorBoard, matplotlib
- **Deployment:** Streamlit

## 📈 Results

The model achieves high accuracy in predicting customer churn. Use TensorBoard to visualize training metrics:

```bash
tensorboard --logdir logs/fit
```

## 📝 Notebooks Overview

| Notebook | Purpose |
|----------|---------|
| `experiments.ipynb` | Train classification model |
| `hyperparametertuningann.ipynb` | Optimize hyperparameters |
| `prediction.ipynb` | Make predictions on new data |
| `salaryregression.ipynb` | Train regression model |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


---

⭐ **Star this repo if you find it helpful!**