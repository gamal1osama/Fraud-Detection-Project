import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 100
from typing import Union, Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report , precision_recall_curve ,confusion_matrix , auc


def get_auc_precision_recall(
        y_pred_prob: Union[np.ndarray, pd.Series, list],
        y_true: Union[np.ndarray, pd.Series, list]
    ) -> float:
    """
    Compute the area under the Precision-Recall (PR) curve.

    Args:
        y_pred_prob: Predicted probabilities or scores for the positive class
        y_true: True binary labels

    Returns:
        float: Area under the Precision-Recall curve
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1)")

    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)
    area_under_PRcurve = float(auc(x=recall, y=precision))

    return area_under_PRcurve

def classification_report_with_cm(  
        y_true: Union[np.ndarray, pd.Series, list],
        y_pred: Union[np.ndarray, pd.Series, list],
        save_png: bool = False,
        path: Optional[str] = ".",
        title: str = "Confusion Matrix"
        ) -> Dict[str, Any]:
    """
    Print classification report and plot a binary confusion matrix.
    
    Args:
        y_true: True labels (0/1)
        y_pred: Predicted labels (0/1)
        save_png: Whether to save the plot as PNG
        path: Directory to save the plot if save_png is True
        title: Title for the report and plot
    
    Returns:
        report_stats: Classification report as a dictionary
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. "
                        f"Got {len(y_true)} and {len(y_pred)}")
    
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1)")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("y_pred must be binary (0/1)")
    
    print(f'{title} Classification Report')
    print(classification_report(y_pred=y_pred, y_true=y_true, digits=5))   
    report_stats = classification_report(y_pred=y_pred, y_true=y_true, digits=5, output_dict=True)
    
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    flatten_cm = cm.flatten()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Reds', fmt='d')

    for i, txt in enumerate(flatten_cm):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, f"{labels[i]}\n{txt}",
                 ha='center', va='center', color='black')

    plt.title(f'Confusion Matrix {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    if save_png:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{title} Confusion Matrix.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
    
    return report_stats

def predict_using_threshold(
        model: BaseEstimator,
        x: pd.DataFrame,
        threshold: float = 0.5
        ) -> np.ndarray:
    """
    Make predictions using a custom probability threshold.

    This function is useful for imbalanced classification problems where
    the default 0.5 threshold may not be optimal.

    Args:
        model: A fitted sklearn classifier with predict_proba method
        x: Input features (numpy array or DataFrame)
        threshold: Probability threshold for positive class classification.
                  Default is 0.5.

    Returns:
        Binary predictions (0 or 1) based on the custom threshold

    Raises:
        AttributeError: If model doesn't have predict_proba method
    """

    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model must have predict_proba method")
    
    y_pred_proba = model.predict_proba(x)  
    y_pred = (y_pred_proba[:, 1] >= threshold).astype('int')
    return y_pred

def precision_recall_thresholds_curve(
        y_pred_prob: Union[np.ndarray, pd.Series, list],
        y_true: Union[np.ndarray, pd.Series, list],
        title: str ="",
        save_png: bool =False,
        path: str = "."
        ) -> None:
    """
    Plot Precision and Recall as functions of the decision threshold.

    Args:
        y_pred_prob: Predicted probabilities or scores for the positive class
        y_true: True binary labels (0/1)
        title: Title of the plot
        save_png: Whether to save the plot as PNG
        path: Directory to save the plot if save_png is True
    """

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1)")
    
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', marker='.')
    plt.plot(thresholds, recall[:-1], label='Recall', marker='.')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision and Recall for different Thresholds {title}')
    plt.legend()

    if save_png:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{title} precision_recall_thresholds_curve.png',  bbox_inches='tight')
        plt.close()
    else:    
        plt.show()
        plt.close()

def plot_pr_curve(
        y_pred_prob: Union[np.ndarray, pd.Series, list],
        y_true: Union[np.ndarray, pd.Series, list],
        title: str ="",
        save_png: bool =False,
        path: str = "."
        ) -> None:
    """
    Plot a Precision-Recall (PR) curve for binary classification.

    Parameters
    ----------
    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores.
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    title : str, optional
        Title of the plot (default: "").
    save_png : bool, optional
        If True, save the figure as a PNG file.
    path : str, optional
        Directory path to save the plot (default: current directory)
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0/1)")
    
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} Precision-Recall Curve')
    plt.legend()

    if save_png:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/{title} precision recall curve.png',  bbox_inches='tight')
        plt.close()
    else:    
        plt.show()
        plt.close()

def get_best_threshold(
        y_pred_prob: Union[np.ndarray, pd.Series, list],
        y_true: Union[np.ndarray, pd.Series, list],
        with_respect_to: str ="f1_score"
        ) -> Tuple[float, np.ndarray]:
    """
    """
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred_prob)
    f1_scores = ((2 * precision * recall) / (precision + recall))

    f1_scores = f1_scores[:-1]   
    precision = precision[:-1]
    recall = recall[:-1]

    threshold_strategies = {
    "f1_score": lambda f1_scores, precision, recall: np.argmax(f1_scores),
    "precision": lambda f1_scores, precision, recall: np.argmax(precision),
    "recall": lambda f1_scores, precision, recall: np.argmax(recall)
    }

    if with_respect_to in threshold_strategies:
        optimal_threshold_index = threshold_strategies[with_respect_to](f1_scores, precision, recall)
    else:
        raise ValueError(f"Invalid value for with_respect_to. Choose from {list(threshold_strategies.keys())}")
    
    optimal_threshold, optimal_f1  = thresholds[optimal_threshold_index], f1_scores[optimal_threshold_index]
    print(f"Optimal Threshold: {optimal_threshold } F1 Score: {optimal_f1}")

    return optimal_threshold , f1_scores

def update_model_stats(
        model_comparison: Dict[str, Dict[str, float]],
        model_name: str,
        report_val: Dict[str, Dict[str, float]],
        metric_config: Optional[Dict[str, Union[Dict[str, bool], bool]]] = None
    ) -> Dict[str, Dict[str, float]]:
    """
    Update model_comparison dictionary with evaluation metrics of the validation set.

    Parameters
    ----------
    model_comparison : dict
        Dictionary storing all models’ evaluation metrics.
    model_name : str
        Name of the model to update.
    report_val : dict
        Output of `classification_report(y_true, y_pred, output_dict=True)`.
    metric_config : dict, optional
        Configuration to specify which metrics to save. Example:
        {
            "pos": {"precision": True, "recall": True, "f1-score": True},
            "neg": {"precision": True, "recall": False, "f1-score": True},
            "macro_avg": True
        }

    Returns
    -------
    dict
        Updated model_comparison dictionary with metrics for the given model.
    """
    if metric_config is None or len(metric_config.keys()) == 0: # Default metrics
        model_comparison[model_name] = {
            "F1 Score Positive class": report_val['1']['f1-score'],
            "F1 Score Negative class": report_val['0']['f1-score'],
            "Precision Positive class": report_val['1']['precision'],
            "Recall Positive class": report_val['1']['recall'],   
            "F1 Score Average": report_val['macro avg']['f1-score'],
            }
    else: # Custom metrics based on metric_config
       model_comparison[model_name] = {}
       for key, value in metric_config['pos'].items():
            if value:
                 model_comparison[model_name][f"{key} positive class"] = report_val['1'][key]

       for key, value in metric_config['neg'].items():
            if value:
                  model_comparison[model_name][f"{key} negative class"] = report_val['0'][key]

       if metric_config['macro_avg']: 
           model_comparison[model_name]['F1 macro avg'] = report_val['macro avg']['f1-score']
                   
    return model_comparison

def evaluate_model(
        model,
        model_comparison: Dict[str, Dict[str, float]],
        path: str,
        title: str,
        X_train,
        y_train,
        X_val,
        y_val,
        eval_config: Dict
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Evaluate a classification model on training and validation sets, update metrics,
    generate plots, and optionally compute optimal threshold.

    Parameters
    ----------
    model : sklearn-like model
        Fitted model with `predict` and `predict_proba` methods.
    model_comparison : dict
        Dictionary storing evaluation metrics for multiple models.
    path : str
        Base path for saving plots.
    title : str
        Name/title of the model.
    X_train, y_train : array-like
        Training data and labels.
    X_val, y_val : array-like
        Validation data and labels.
    eval_config : dict
        Dictionary specifying evaluation settings:
        - 'train': bool, whether to evaluate on training data
        - 'validation': bool, whether to evaluate on validation data
        - 'confusion_matrix': bool, whether to save confusion matrix plots
        - 'precision_recall_threshold': bool, whether to save precision-recall vs threshold plots
        - 'precision_recall': bool, whether to save precision-recall curve
        - 'optimal_threshold': bool, whether to compute optimal threshold
        - 'metric': dict specifying which metrics to save (e.g., 'PR_AUC': True)

    Returns
    -------
    Tuple
        - Updated model_comparison dictionary
        - Optimal threshold computed from training set (float)
    """
    optim_threshold = 0.5
    
    plots_path = path + eval_config['plot_path']
    os.makedirs(plots_path, exist_ok=True)

    save_cm_plots = eval_config['confusion_matrix']
    save_PR_with_thresholds = eval_config['precision_recall_threshold']
    save_PR_curve = eval_config['precision_recall']

    if eval_config['train']:
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:,1]

        classification_report_with_cm(y_train, y_train_pred, save_cm_plots, plots_path, title + ' train')
        precision_recall_thresholds_curve(y_train_pred_proba, y_train, title + ' train', save_PR_with_thresholds, plots_path)
        plot_pr_curve(y_train_pred_proba, y_train, title + ' train', save_PR_curve, plots_path)

    if eval_config['validation']:
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:,1]

        report_val = classification_report_with_cm(y_val, y_val_pred, save_cm_plots, plots_path, title + ' validation')
        plot_pr_curve(y_val_pred_proba, y_val, title + ' validation', save_PR_curve, plots_path)
        
        model_comparison = update_model_stats(model_comparison, title,  report_val, eval_config['metric'])
        
        if eval_config['metric']['PR_AUC']:
            model_comparison[title]['PR AUC'] = get_auc_precision_recall(y_val_pred_proba, y_val)

        if eval_config['optimal_threshold']: 
            optim_threshold , _ = get_best_threshold(y_train_pred_proba, y_train)
            
            y_val_pred = predict_using_threshold(model, X_val, optim_threshold)

            report_val = classification_report_with_cm(y_val, y_val_pred, save_cm_plots, plots_path, title + ' val with optimal threshold')
            model_comparison = update_model_stats(model_comparison, title + ' optimal threshold', report_val, eval_config['metric'])
            

    return model_comparison, optim_threshold
