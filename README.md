# Fraud Detection Project

## Project Overview

This project implements a comprehensive fraud detection system using various machine learning models. The primary goal is to accurately identify fraudulent transactions in a given dataset, minimizing false positives while maximizing the detection of actual fraud. A significant challenge addressed in this project is the **highly imbalanced nature of fraud datasets**, where fraudulent transactions are extremely rare compared to legitimate ones. The system is designed to be robust, scalable, and easy to deploy, leveraging best practices in data preprocessing, model training, and evaluation to effectively handle this imbalance.

Fraudulent activities pose significant financial risks to individuals and institutions. Early and accurate detection is crucial to mitigate these risks. This repository provides a complete pipeline, from data loading and preprocessing to model training, evaluation, and comparison, offering a robust solution for tackling this challenge.

## Features

- **Data Preprocessing**: Includes robust scaling and various data balancing techniques (e.g., SMOTE, RandomOverSampler, RandomUnderSampler, SMOTEENN, SMOTETomek) to handle imbalanced datasets common in fraud detection.
- **Multiple Machine Learning Models**: Implements and evaluates several classification algorithms, including:
    - Random Forest
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Neural Networks (MLPClassifier)
    - Ensemble Voting Classifier
- **Hyperparameter Optimization**: Utilizes RandomizedSearchCV and GridSearchCV for efficient hyperparameter tuning to achieve optimal model performance.
- **Comprehensive Evaluation Metrics**: Models are evaluated using a range of metrics, including F1-score, precision, recall, confusion matrices, precision-recall curves, and PR AUC, with a focus on the positive class (fraudulent transactions).
- **Optimal Threshold Selection**: Provides functionality to determine and apply optimal classification thresholds based on F1-score, precision, or recall to further enhance model performance on imbalanced data.
- **Configurable Pipeline**: Training and data preprocessing steps are highly configurable via YAML files, allowing for easy experimentation and adaptation to different datasets or requirements.
- **Model Persistence**: Trained models and evaluation results are saved for future use and analysis.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/gamal1osama/Fraud-Detection-Project.git
   cd Fraud-Detection-Project
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



## Usage

### Data Preparation

The dataset used in this project is the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Ensure your dataset is split into `train.csv`, `val.csv`, and `test.csv` and placed in the `data/` directory. The `configrations/data_configrations.yml` file should be updated to reflect the correct paths and target column name.

### Training Models

To train the models, run the `train.py` script. You can specify configuration files for data and training parameters:

```bash
python train.py --config configrations/data_configrations.yml --training configrations/training_configration.yml
```

The `configrations/training_configration.yml` file allows you to enable/disable training for specific models, configure hyperparameter search settings (RandomizedSearchCV/GridSearchCV), and define evaluation preferences.

### Evaluation

During training, models are evaluated on both training and validation datasets. Confusion matrices, precision-recall curves, and precision-recall vs. threshold plots are generated and saved in the `models/<timestamp>/evaluation/plots/` directory. A `model_comparison-(validation dataset).png` heatmap is also generated, summarizing key performance metrics across all trained models.

## Model Performance

The project evaluates several machine learning models for fraud detection. The performance is summarized in the following comparison chart, focusing on key metrics on the validation dataset:

![Model Comparison - Validation Dataset](https://github.com/gamal1osama/Fraud-Detection-Project/blob/main/models/2025_09_25_18_09/model_comparison-(validation%20dataset).png)

As observed from the validation dataset, the **Random Forest** model achieved the highest F1-score macro average of **93%**, demonstrating its superior ability to balance precision and recall across both positive (fraud) and negative (non-fraud) classes. This makes Random Forest the most effective model for this fraud detection task among those evaluated.

## Project Structure

```
Fraud-Detection-Project/
├── EDA.ipynb                 # Exploratory Data Analysis notebook
├── configrations/
│   ├── data_configrations.yml  # Configuration for data loading and preprocessing
│   └── training_configration.yml # Configuration for model training and evaluation
├── data/
│   ├── creditcard.csv        # Raw dataset (example)
│   ├── test.csv              # Test dataset
│   ├── train.csv             # Training dataset
│   └── val.csv               # Validation dataset
├── models/
│   └── <timestamp>/
│       ├── evaluation/       # Directory for evaluation plots
│       │   └── plots/
│       │       └── *.png     # Various plots (Confusion Matrix, PR curves, etc.)
│       ├── model_comparison-(validation dataset).png # Heatmap comparing model performance
│       └── trained_models.pkl # Serialized trained models
├── my_data_profile.html      # Data profiling report
├── requirements.txt          # Python dependencies and versions
├── train.py                  # Main script for training and evaluating models
└── utils/
    ├── __pycache__/
    ├── data_utils.py         # Utility functions for data loading, scaling, and balancing
    ├── eval_utils.py         # Utility functions for model evaluation and plotting
    └── model_utils.py        # Utility functions for loading configurations and saving models
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. (Note: A `LICENSE` file is assumed. If not present, consider adding one.)

## Contact

For any questions or inquiries, please contact [Your Name/Email/GitHub Profile].


