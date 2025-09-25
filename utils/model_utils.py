import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from typing import Dict, Any


def load_model(path: str) -> object | None:
    """
    Load a model from a given path using joblib.
    Args:
        path (str): Path to the saved model file.
    Returns:
        object | None: Loaded model, or None if loading failed.
    """
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_model_comparison(
        model_comparison: Dict[str, Dict[str, float]],
        path: str
        ) -> None:
    """
    Save a heatmap of model comparison metrics.

    Args:
        model_comparison (Dict[str, Dict[str, float]]): Nested dictionary of model performance metrics.
        path (str): Path (including filename) to save the heatmap PNG.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "model_comparison.png")

    model_comparison = pd.DataFrame(model_comparison).T
    
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1.2)
    
    sns.heatmap(model_comparison, annot=True, cmap='viridis', cbar=True, annot_kws={"size": 12}, fmt='.2f')
    
    plt.title('Model Performance Comparison', fontsize=20, pad=20)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def load_config(config_path: str) -> Dict[str, Any] | None:
    """
    Load a YAML configuration file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a Python dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    