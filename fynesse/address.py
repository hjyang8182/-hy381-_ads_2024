# This file contains code for suporting addressing questions in the data

"""Here are some of the imports we might expect 
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
import warnings
from .assess import *
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

def find_all_oa_student_features(conn, oa_id): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    query = f"""
    select poi_count.*, ns.L15 from oa_poi_count_data as poi_count inner join ns_sec_boundary_data as ns on poi_count.oa_id = ns.oa_id where poi_count.oa_id = '{oa_id}'
    """
    cur.execute(query)
    all_poi_data = cur.fetchall()
    all_poi_data_df = pd.DataFrame(all_poi_data)
    all_poi_data_df.set_index('oa_id')
    return all_poi_data_df

def find_all_oa_toddler_features(conn, oa_id):
    cur = conn.cursor(pymysql.cursors.DictCursor)
    query = f"""
    select * from toddler_data where oa_id = '{oa_id}'
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

# def k_fold_cross_validation_predict_students_regularized(k, dataset, feature_cols, alpha, l1_wt):
#     performances = []
#     model_coefs = []
#     k_folds = split_dataset(k, dataset)
#     for i in range(k): 
#         # Prepare test data
#         test_data = k_folds[i]
#         test_indices = test_data.index.values
#         test_features = test_data.loc[test_indices][feature_cols].values
#         test_labels = test_data.loc[test_indices]['L15'].values
        
#         # Train on traindata 
#         train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
#         train_indices = train_data.index.values
#         train_features = train_data.loc[train_indices][feature_cols].values
#         train_labels = train_data.loc[train_indices]['L15'].values
    
#         m_linear = sm.OLS(train_labels, train_features)
#         results_linear = m_linear.fit_regularized(alpha = alpha, L1_wt = l1_wt)
#         # if coefficients are largely different, overfitting 
#         model_coefs.append(results_linear.params)

#         train_pred = results_linear.predict(train_features)
#         train_performance = measure_performance(train_pred, train_labels)
#         # Test on test data
#         test_pred = results_linear.predict(test_features)
#         test_performance = measure_performance(test_pred, test_labels)
#         performances.append((train_performance,test_performance))
#     return performances, model_coefs


# def k_fold_cross_validation_predict_students(k, dataset, feature_cols):
#     performances = []
#     model_coefs = []
#     k_folds = split_dataset(k, dataset)
#     for i in range(k): 
#         # Prepare test data
#         test_data = k_folds[i]
#         test_indices = test_data.index.values
#         test_features = test_data.loc[test_indices][feature_cols].values
#         test_labels = test_data.loc[test_indices]['L15'].values
        
#         # Train on traindata 
#         train_data = pd.DataFrame(pd.concat([k_folds[j] for j in range(k) if j != i]))
#         train_indices = train_data.index.values
#         train_features = train_data.loc[train_indices][feature_cols].values
#         train_labels = train_data.loc[train_indices]['L15'].values
    
#         m_linear = sm.OLS(train_labels, train_features)
#         results_linear = m_linear.fit()
#         # if coefficients are largely different, overfitting 
#         model_coefs.append(results_linear.params)

#         train_pred = results_linear.get_prediction(train_features).summary_frame()
#         train_performance = measure_performance(train_pred['mean'], train_labels)
#         # Test on test data
#         test_pred = results_linear.get_prediction(test_features).summary_frame()
#         test_performance = measure_performance(test_pred['mean'], test_labels)
#         performances.append((train_performance,test_performance))
#     return performances, model_coefs

def plot_regularized_model_performance(all_features, label_col, feature_cols):
    linear_k_fold_results_regularized_l1 =k_fold_cross_validation_regularized(10, all_features, label_col, feature_cols, 0.1, 0)
    linear_k_fold_perf_regularized_l1 = linear_k_fold_results_regularized_l1[0]
    linear_k_fold_coefs_regularized_l1 = linear_k_fold_results_regularized_l1[1]

    train_perf_l1 = np.array(list(map(lambda x : x[0], linear_k_fold_perf_regularized_l1)))
    test_perf_l1 = np.array(list(map(lambda x: x[1], linear_k_fold_perf_regularized_l1)))
    k_vals = np.arange(100)
    plt.scatter(k_vals, train_perf_l1, color = 'blue')
    plt.scatter(k_vals, test_perf_l1, color = 'red')
    plt.xlabel('K value')
    plt.ylabel('Root Mean Sqared Error')
    print(f"Train RMSE average: {np.mean(train_perf_l1)}")
    print(f"Test RMSE average: {np.mean(test_perf_l1)}" )
def fit_linear_model_regularized(label_col, feature_cols, all_poi_data_df, alpha=0.0002, l1_wt=1.0): 
    all_features = all_poi_data_df[feature_cols].values.astype(float)
    population = all_poi_data_df[label_col].values.astype(float)

    m_linear_all_feat = sm.OLS(population, all_features)
    results_linear = m_linear_all_feat.fit_regularized(alpha=alpha, L1_wt = l1_wt)
    return results_linear.params

# TASK 2
def fit_final_model(conn, lad_ids, transport_type, lad_boundaries, num_lsoas):
    all_feature_dfs = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        for lad_id in lad_ids: 
            # conn, lad_id, transport_type, lad_boundaries, num_lsoas = 5
            feature_df = find_all_features_with_house_types(conn, lad_id, transport_type, lad_boundaries, num_lsoas)
            all_feature_dfs.append(feature_df) 
        all_features = pd.concat(all_feature_dfs)
        all_features = all_features.dropna()   
        columns = ['transport_usage', 'car_availability', 'avg_dist', 'detached', 'semi_detached', 'terraced', 'flats', 'new_build']
        label_col = 'pct_inc'
        features = all_features[columns].values.astype(float)
        labels = all_features[label_col].values.astype(float)
        m_linear_all_feat_tube = sm.OLS(labels, features)
        results_linear_tube = m_linear_all_feat_tube.fit()
    return all_features, results_linear_tube

def get_test_features(conn, lad_ids, transport_type, lad_boundaries, num_lsoas):
    test_features_dfs = []
    for lad_id in lad_ids: 
        feature_df = find_all_features_with_house_types(conn, lad_id, transport_type, lad_boundaries, num_lsoas)
        test_features_dfs.append(feature_df) 
    rail_test_features_df = pd.concat(test_features_dfs)
    return rail_test_features_df


def make_prediction(conn, lad_to_region, input_feature, transport_type, lad_boundaries, num_lads, num_lsoas):
    lad_to_region_by_rgn = lad_to_region.groupby('RGN21CD')['LAD21CD'].sample(n=num_lads)
    random_lads = lad_to_region_by_rgn.values
    all_features, result_linear = fit_final_model(conn, random_lads, transport_type, lad_boundaries, num_lsoas)
    params = result_linear.params
    pred = np.sum(params * input_feature)
    return pred