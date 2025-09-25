import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional, Dict, Any, Union
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def balance_the_data(
        X_train: np.ndarray,
        y_train: pd.Series,
        type_of_sampling: str = 'smote',
        sampling_strategy: Union[str, float] = 'auto',
        k: int = 5,
        random_state: int = 42,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance imbalanced dataset using various sampling techniques.
    
    Args:
        X_train: Training features as numpy array
        y_train: Training target as pandas Series
        type_of_sampling: Type of sampler ('under', 'over', 'smote', 'SMOTEENN', 'SMOTETomek')
        sampling_strategy: Sampling strategy ('auto', float ratio, etc.)
        k: Number of neighbors for SMOTE-based methods
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X and y as numpy arrays
    """
    samplers = {
        'under': RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state),
        'over': RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state),
        'smote': SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k),
        'SMOTEENN': SMOTEENN(
            random_state=random_state, 
            sampling_strategy=sampling_strategy,
            smote=SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k)
        ),
        'SMOTETomek': SMOTETomek(
            random_state=random_state, 
            sampling_strategy=sampling_strategy,
            smote=SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k),
            n_jobs=-1
        )
    }

    if type_of_sampling not in samplers:
        raise ValueError(f"Invalid balance type: {type_of_sampling}. Choose from {list(samplers.keys())}")
    
    y_array = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
    
    if verbose:
        print("Dataset before balancing:")
        print(f"Class 0 (Non-fraud): {np.sum(y_array == 0):,}")
        print(f"Class 1 (Fraud):     {np.sum(y_array == 1):,}")
        print(f"Total samples:       {len(y_array):,}")

    X_resampled, y_resampled = samplers[type_of_sampling].fit_resample(X_train, y_array)

    if verbose:
        print("\nDataset after balancing:")
        print(f"Class 0 (Non-fraud): {np.sum(y_resampled == 0):,}")
        print(f"Class 1 (Fraud):     {np.sum(y_resampled == 1):,}")
        print(f"Total samples:       {len(y_resampled):,}")

    return X_resampled, y_resampled

def load_train_val(
        data_configurations: Dict[str, Any],
        ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load training and validation datasets from CSV files specified in configuration.

    Reads train and validation data from the file paths defined in the configuration,
    then splits each dataset into features (X) and target (y) based on the specified target column.

    Args:
        data_configurations (Dict[str, Any]): Configuration dictionary containing dataset paths 
            and target column name. Expected structure:
            {
                'dataset': {
                    'train': {'path': 'path/to/train.csv'},
                    'val': {'path': 'path/to/val.csv'},
                    'target': 'name_of_target_column'
                }
            }

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
            - X_train (pd.DataFrame): Training features
            - y_train (pd.Series): Training target values
            - X_val (pd.DataFrame): Validation features  
            - y_val (pd.Series): Validation target values
    """
    train = pd.read_csv(data_configurations['dataset']['train']['path'])
    val = pd.read_csv(data_configurations['dataset']['val']['path'])

    X_train = train.drop(data_configurations['dataset']['target'], axis=1)
    y_train = train[data_configurations['dataset']['target']]

    X_val = val.drop(data_configurations['dataset']['target'], axis=1)
    y_val = val[data_configurations['dataset']['target']]

    return X_train, y_train, X_val, y_val
    
def load_test(
        Data_Configurations: Dict[str, Any],
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test dataset from CSV file specified in configuration.

    Reads test data from the file path defined in the configuration,
    then splits it into features (X) and target (y) based on the specified target column.

    Args:
        Data_Configurations (Dict[str, Any]): Configuration dictionary containing dataset path 
            and target column name. Expected structure:
            {
                'dataset': {
                    'test': {'path': 'path/to/test.csv'},
                    'target': 'name_of_target_column'
                }
            }

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X_test (pd.DataFrame): Test features
            - y_test (pd.Series): Test target values
    """
    test = pd.read_csv(Data_Configurations['dataset']['test']['path'])

    X_test = test.drop(Data_Configurations['dataset']['target'], axis=1)
    y_test = test[Data_Configurations['dataset']['target']]

    return X_test, y_test

def scaling_train_and_val(
        train: pd.DataFrame,
        val: Optional[pd.DataFrame] = None,
        type_of_scaling: str = 'robust',
        ) -> Tuple[np.ndarray, Optional[np.ndarray], Union[StandardScaler, MinMaxScaler, RobustScaler]]:
    """
    Convert DataFrame(s) to NumPy and scale using the specified scaler.

    Args:
        train: Training data as DataFrame .
        val: Validation data as DataFrame. Defaults to None.
        type_of_scaling: Type of scaler to use. Options: 'minmax', 'standard', 'robust'. 
                        Defaults to 'robust'.

    Returns:
        Tuple containing:
            - Scaled training data as numpy array
            - Scaled validation data as numpy array if val is provided, else None
            - The fitted scaler (for later use)

    Raises:
        ValueError: If invalid scaler type is provided or inputs are not DataFrames.
    """
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }

    if type_of_scaling not in scalers:
        raise ValueError(
            f"Invalid scaler type: {type_of_scaling}. "
            f"Choose from {list(scalers.keys())}"
        )
    
    if not isinstance(train, pd.DataFrame):
        raise ValueError(f"train must be a DataFrame, got {type(train)}")
    
    if val is not None and not isinstance(val, pd.DataFrame):
        raise ValueError(f"val must be a DataFrame or None, got {type(val)}")
    
    scaler = scalers[type_of_scaling]
    train_array  = train.to_numpy()
    val_array  = val.to_numpy() if val is not None else None
    
    scaled_train = scaler.fit_transform(train_array)
    scaled_val = scaler.transform(val_array) if val_array is not None else None

    return scaled_train, scaled_val, scaler
