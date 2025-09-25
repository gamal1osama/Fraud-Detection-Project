import os
import joblib
import argparse
import datetime

import numpy as np
import pandas as pd

from utils.eval_utils import *
from utils.data_utils import load_train_val, scaling_train_and_val, balance_the_data
from utils.model_utils import load_config, save_model_comparison

from mlxtend.classifier import EnsembleVoteClassifier 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , StratifiedKFold , RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

def train_random_forest(
        X_train,
        y_train,
        X_val,
        y_val,
        model_comparison,
        training
        ) :
    """
    """
    if training['trainer']['Random_forest']['Randomized_Search']:
        param_distributions = {
            'n_estimators': [200, 400, 600 ,800],
            'min_samples_leaf': [2, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'class_weight': [{0: 0.20, 1: 0.80}, 'balanced_subsample', {0: 0.15, 1: 0.85}],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_jobs=-1, bootstrap=True, random_state=42),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=42 
        )

        random_search.fit(X_train, y_train)

        parameters = random_search.best_params_
        print("Best Hyperparameters for Random Forest:", parameters)
    else:
        parameters = training['trainer']['Random_forest']['parameters']


    rf = RandomForestClassifier(
        **parameters,         
        random_state=42  
    )

    rf.fit(X_train, y_train) 

    model_comparison , optimal_threshold = evaluate_model(rf, model_comparison, path, 'Random Forest', X_train, y_train, X_val, y_val, training['evaluation'])

    return {"model": rf ,  "parameters": parameters, "threshold": optimal_threshold} 


def train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, model_comparison,  training):

    best_params = {}

    if training['trainer']['Logistic_Regression']['grid_search'] == True:
             param_grid = {
                            'C':            [0.1, 1.0, 10.0],
                            'penalty':      ['l2'],
                            'class_weight': ['balanced', None, {0: 0.35, 1: 0.65}, {0: 0.25, 1: 0.75}, {0: 0.15, 1: 0.85}],
                            'solver':       ['sag', 'lbfgs', 'saga', ' newton-cg'],  
                            'max_iter':     [400, 500, 600, 800],
                        }
                        
             lr = LogisticRegression()
             scorer = make_scorer(f1_score, pos_label=1)

             stratified_kfold = StratifiedKFold(n_splits=5, 
                                            shuffle=True,
                                            random_state=42)

             grid_search = GridSearchCV(lr, 
                                    param_grid,cv=stratified_kfold, 
                                    scoring=scorer, 
                                    n_jobs=-1)

             grid_search.fit(X_train_scaled, y_train)

             best_params = grid_search.best_params_
             print("Best Hyperparameters:", best_params)

  
    else:
        best_params = training['trainer']['Logistic_Regression']['parameters']
       

    lr = LogisticRegression(**best_params, random_state=42)
    lr.fit(X_train_scaled, y_train)

    model_comparison , optimal_threshold = evaluate_model(lr, model_comparison, path, 'Logistic Regression', X_train_scaled, y_train, X_val_scaled, y_val, training['evaluation'])

    return {"model": lr , "parameters": best_params, "threshold": optimal_threshold}


def train_knn(X_train, y_train, X_val, y_val, model_comparison, training):

    if training['trainer']['KNN']['grid_search'] == True:
        param_distributions = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }

        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, pos_label=1)

        random_search = RandomizedSearchCV(
            estimator=KNeighborsClassifier(n_jobs=-1),
            param_distributions=param_distributions,
            scoring=scorer,
            cv=stratified_kfold,
            n_iter=20,  
            n_jobs=-1,
            verbose=2,
            random_state=42 
        )

        random_search.fit(X_train, y_train)

        parameters = random_search.best_params_
        print("Best Hyperparameters for KNN:", parameters)
    else:
        parameters = training['trainer']['KNN']['parameters']


    knn = KNeighborsClassifier(**parameters, n_jobs=-1)

    knn.fit(X_train, y_train)

    model_comparison , optimal_threshold = evaluate_model(knn, model_comparison, path, 'KNN', X_train, y_train, X_val, y_val, training['evaluation'])

    return {"model": knn , "parameters": parameters, "threshold": optimal_threshold}

def train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, model_comparison,  training):

    best_params = {}

    if training['trainer']['Neural_Network']['Randomized_Search'] == True:
        param_dist = {
        'activation': ['relu'],
        'hidden_layer_sizes': [
            (30, 20), 
            (30, 20, 10), 
            (40, 30, 20), 
            (64, 32, 16),
            (64, 32, 32, 16)
        ],
        'solver': ['adam', 'sgd'],
        'batch_size': [64, 128, 512],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.001, 0.01, 0.025],
        'max_iter': [500, 800, 1000, 2000],
        'random_state': [42]
        }

        MLP_CV = MLPClassifier()
        scorer = make_scorer(f1_score, pos_label=1)  
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(MLP_CV, param_distributions=param_dist, n_iter=30, cv=cv, scoring=scorer, n_jobs=-1, random_state=42)
        random_search.fit(X_train_scaled, y_train)

        MLP = MLPClassifier(**best_params)

    else:
        # load parameters from config file
        best_params = training['trainer']['Neural_Network']['parameters']
    
        MLP = MLPClassifier(
            hidden_layer_sizes=eval(best_params['hidden_layer_sizes']), 
            activation=best_params['activation'],
            solver=best_params['solver'],
            alpha=best_params['alpha'],
            batch_size=best_params['batch_size'],
            learning_rate_init=best_params['learning_rate_init'],
            max_iter=best_params['max_iter'],
            random_state=42
        )

    MLP.fit(X_train_scaled, y_train)

    model_comparison, optimal_threshold = evaluate_model(MLP, model_comparison, path, 'Neural Network', X_train_scaled, y_train, X_val_scaled, y_val, training['evaluation'])

    return {"model": MLP ,  "parameters": best_params, "threshold": optimal_threshold} 


def train_voting_classifier(X_train, y_train, x_val, y_val, models, model_comparison, training):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # make scaler learn statistics from training data
    param = training['trainer']['Voting_Classifier']['parameters']

    # Ensure models are present in the provided dictionary
    required_models = ['Logistic_Regression', 'Neural_Network', 'Random_forest']
    missing_models = [model for model in required_models if model not in models]
    
    if missing_models:
        raise ValueError(f"The following required models are missing: {', '.join(missing_models)}")

    try:
        voting_classifier = EnsembleVoteClassifier(
            clfs=[
                make_pipeline(scaler, models['Logistic_Regression']['model']),
                make_pipeline(scaler, models['Neural_Network']['model']),
                models['Random_forest']['model'],
            ],
            weights=param['weights'],
            fit_base_estimators=param['fit_base_estimators'],
            use_clones=param['use_clones'],
            voting=param['voting'],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the voting classifier: {e}")


    voting_classifier.fit(X_train, y_train) #  no refiting required here

    model_comparison, optimal_threshold = evaluate_model(voting_classifier, model_comparison, path, 'Voting Classifier', X_train, y_train, x_val, y_val, training['evaluation'])

    return {"model": voting_classifier , "parameters": param, "threshold": optimal_threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file of dataset and preprocessing", default="configrations/data_configrations.yml")
    parser.add_argument("--training", help="config file of training and evaluation", default="configrations/training_configration.yml")
    args = parser.parse_args()

    config = load_config(args.config)
    training = load_config(args.training)

    np.random.seed(42)

    X_train, y_train, X_val, y_val = load_train_val(config)
    X_train_original, y_train_original = X_train.copy(), y_train.copy()

    X_train_scaled, X_val_scaled, _ = scaling_train_and_val(X_train, X_val, config['preprocessing']['scaler_type'])

    if config['balancing']['do_balance']: 
        X_train_scaled_balanced, y_train_balanced = balance_the_data(
            X_train_scaled, y_train, 
            type_of_sampling=config['balancing']['method'], 
            sampling_strategy=config['balancing']['sampling_strategy'], 
            k=5,  
            random_state=42
        )
        
        X_train_original_balanced, y_train_balanced_original = balance_the_data(
            X_train, y_train,
            type_of_sampling=config['balancing']['method'],
            sampling_strategy=config['balancing']['sampling_strategy'], 
            k=5,
            random_state=42
        )
        
        X_train_scaled = X_train_scaled_balanced
        X_train = X_train_original_balanced
        y_train = y_train_balanced  
    
    model_comparison, models = {}, {}

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    path = f'models/{now}/'

    os.makedirs(path, exist_ok=True)

    if training['trainer']['Random_forest']['train']:
        models["Random_forest"] = train_random_forest(X_train, y_train, X_val, y_val, model_comparison, training)
    
    if training['trainer']['Logistic_Regression']['train']:
        models['Logistic_Regression'] = train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, model_comparison, training)
    
    if training['trainer']['KNN']['train']:
        models['KNN'] = train_knn(X_train_scaled, y_train, X_val_scaled, y_val, model_comparison, training)

    if training['trainer']['Neural_Network']['train']:
        models['Neural_Network'] = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, model_comparison, training)    

    if training['trainer']['Voting_Classifier']['train']:
        models['Voting_Classifier'] = train_voting_classifier(X_train, y_train, X_val, y_val, models, model_comparison, training)


    model_path = path + "trained_models.pkl"
    joblib.dump(models, model_path)
    print('Model saved at: {}'.format(model_path))
    print('Evaluation plots saved at: {}evaluation/plot'.format(path))
    
    if  model_comparison:
        # Save the model comparison
        model_comparison_path = path + "model_comparison-(validation dataset).png"
        save_model_comparison(model_comparison, model_comparison_path)

        print('\nModels comparison:\n')
        print(pd.DataFrame(model_comparison).T.to_markdown())

