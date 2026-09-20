# Breast Cancer Prediction using Machine Learning

## CodeAlpha Internship - Task 4

This project demonstrates classification of breast tumor samples as **malignant** or **benign** using machine-learning algorithms.

> **Important:** This is an educational machine-learning project. It is not a medical diagnostic system and must not be used for clinical decisions.

### Objective
Use structured medical features to build and compare classification models for breast cancer prediction.

### Dataset
The project uses the **Breast Cancer Wisconsin Diagnostic dataset** provided through `scikit-learn`.

It contains:
- 569 samples
- 30 numerical input features
- 2 target classes: malignant and benign

The dataset is included with scikit-learn, so no manual CSV download is required.

### Models
The project compares:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

The final model is selected based on the validation ROC-AUC among the evaluated models, without assuming that a single metric is sufficient for medical use.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

### Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

### Project Structure

```text
Task_4_Breast_Cancer_Prediction_ML/
├── src/
│   └── train.py
├── models/
├── results/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### How to Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The training script creates the trained model and evaluation plots inside `models/` and `results/`.

### Features
The application accepts the 30 numerical diagnostic measurements used by the dataset and returns the model's predicted class and probability.

### Responsible Use
The output represents a machine-learning prediction on a public research dataset. It should not be interpreted as a diagnosis or medical advice.
