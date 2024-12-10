# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from .config import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pymysql

def split_dataset(k, dataset):
    return np.array_split(dataset.sample(frac=1, random_state = 30), k)

def measure_performance(y_pred, y): 
    rmse = np.sqrt(np.mean((y-y_pred)**2))
    return rmse

def k_fold_cross_validation(k, dataset, label_col, feature_cols): 
    performances = []
    model_coefs = []
    k_folds = split_dataset(k, dataset)
    for i in range(k): 
        # Prepare test data
        test_data = k_folds[i]
        test_indices = test_data.index.values
        test_features = test_data.loc[test_indices][feature_cols].values
        test_labels = test_data.loc[test_indices][label_col].values
        
        # Train on traindata 
        train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
        train_indices = train_data.index.values
        train_features = train_data.loc[train_indices][feature_cols].values
        train_labels = train_data.loc[train_indices][label_col].values
    
        m_linear = sm.OLS(train_labels, train_features)
        results_linear = m_linear.fit()
        # if coefficients are largely different, overfitting 
        model_coefs.append(results_linear.params)

        train_pred = results_linear.get_prediction(train_features).summary_frame()
        train_performance = measure_performance(train_pred['mean'], train_labels)
        # Test on test data
        test_pred = results_linear.get_prediction(test_features).summary_frame()
        test_performance = measure_performance(test_pred['mean'], test_labels)
        performances.append((train_performance,test_performance))
    return performances, model_coefs

def k_fold_cross_validation_regularized(k, dataset, label_col, feature_cols, alpha, l1_wt):
    performances = []
    model_coefs = []
    k_folds = split_dataset(k, dataset)
    for i in range(k): 
        # Prepare test data
        test_data = k_folds[i]
        test_indices = test_data.index.values
        test_features = test_data.loc[test_indices][feature_cols].values
        test_labels = test_data.loc[test_indices][label_col].values
        
        # Train on traindata 
        train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
        train_indices = train_data.index.values
        train_features = train_data.loc[train_indices][feature_cols].values
        train_labels = train_data.loc[train_indices][label_col].values
    
        m_linear = sm.OLS(train_labels, train_features)
        results_linear = m_linear.fit_regularized(alpha = alpha, L1_wt = l1_wt)
        # if coefficients are largely different, overfitting 
        model_coefs.append(results_linear.params)

        train_pred = results_linear.predict(train_features)
        train_performance = measure_performance(train_pred, train_labels)
        # Test on test data
        test_pred = results_linear.predict(test_features)
        test_performance = measure_performance(test_pred, test_labels)
        performances.append((train_performance,test_performance))
    return performances, model_coefs

def fit_linear_model(poi_data_df, feature_cols, label_col, alpha, l1_wt): 
    features = poi_data_df[feature_cols].values
    labels = poi_data_df[label_col].values
    m_linear_all_feat = sm.OLS(labels, features)
    results_linear = m_linear_all_feat.fit_regularized(alpha = alpha, L1_wt = l1_wt)
    return results_linear.params

def find_oa(connection, latitude, longitude): 
    cur = connection.cursor(pymysql.cursors.DictCursor)
    query = f"""
    select oa_id, ST_Distance(geometry, ST_GeomFromText('POINT({longitude} {latitude})')) as distance 
    from oa_boundary_data 
    ORDER BY distance
    limit 1
    """
    cur.execute(query)
    oa_query = cur.fetchall()[0]
    oa_id = oa_query['oa_id']
    return oa_id

def find_all_oa_features(conn, oa_id): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    query = """
    select poi_count.*, ns.L15 from oa_poi_count_data as poi_count inner join ns_sec_boundary_data as ns on poi_count.oa_id = ns.oa_id
    """
    cur.execute(query)
    all_poi_data = cur.fetchall()
    all_poi_data_df = pd.DataFrame(all_poi_data)
    all_poi_data_df.set_index('oa_id')
    return all_poi_data_df

def split_dataset(k, dataset):
    return np.array_split(dataset.sample(frac=1, random_state = 30), k)

def measure_performance(y_pred, y): 
    rmse = np.sqrt(np.mean((y-y_pred)**2))
    return rmse

def k_fold_cross_validation_predict_students_regularized(k, dataset, feature_cols, alpha, l1_wt):
    performances = []
    model_coefs = []
    k_folds = split_dataset(k, dataset)
    for i in range(k): 
        # Prepare test data
        test_data = k_folds[i]
        test_indices = test_data.index.values
        test_features = test_data.loc[test_indices][feature_cols].values
        test_labels = test_data.loc[test_indices]['L15'].values
        
        # Train on traindata 
        train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
        train_indices = train_data.index.values
        train_features = train_data.loc[train_indices][feature_cols].values
        train_labels = train_data.loc[train_indices]['L15'].values
    
        m_linear = sm.OLS(train_labels, train_features)
        results_linear = m_linear.fit_regularized(alpha = alpha, L1_wt = l1_wt)
        # if coefficients are largely different, overfitting 
        model_coefs.append(results_linear.params)

        train_pred = results_linear.predict(train_features)
        train_performance = measure_performance(train_pred, train_labels)
        # Test on test data
        test_pred = results_linear.predict(test_features)
        test_performance = measure_performance(test_pred, test_labels)
        performances.append((train_performance,test_performance))
    return performances, model_coefs


def k_fold_cross_validation_predict_students(k, dataset, feature_cols):
    performances = []
    model_coefs = []
    k_folds = split_dataset(k, dataset)
    for i in range(k): 
        # Prepare test data
        test_data = k_folds[i]
        test_indices = test_data.index.values
        test_features = test_data.loc[test_indices][feature_cols].values
        test_labels = test_data.loc[test_indices]['L15'].values
        
        # Train on traindata 
        train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
        train_indices = train_data.index.values
        train_features = train_data.loc[train_indices][feature_cols].values
        train_labels = train_data.loc[train_indices]['L15'].values
    
        m_linear = sm.OLS(train_labels, train_features)
        results_linear = m_linear.fit()
        # if coefficients are largely different, overfitting 
        model_coefs.append(results_linear.params)

        train_pred = results_linear.get_prediction(train_features).summary_frame()
        train_performance = measure_performance(train_pred['mean'], train_labels)
        # Test on test data
        test_pred = results_linear.get_prediction(test_features).summary_frame()
        test_performance = measure_performance(test_pred['mean'], test_labels)
        performances.append((train_performance,test_performance))
    return performances, model_coefs


def fit_linear_model_regularized(label_col, feature_cols, all_poi_data_df, alpha=0.0002, l1_wt=1.0): 
    all_features = all_poi_data_df[feature_cols].values.astype(float)
    population = all_poi_data_df[label_col].values.astype(float)

    m_linear_all_feat = sm.OLS(population, all_features)
    results_linear = m_linear_all_feat.fit_regularized(alpha=alpha, L1_wt = l1_wt)
    return results_linear.params