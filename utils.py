import matplotlib.font_manager
import pandas as pd
import glob
import re
import numpy as np
import copy
import pickle
import math
import joblib
import os
import shap
import json
import cupy
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import cudf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, matthews_corrcoef
)
from sklearn.utils.class_weight import compute_class_weight
from PIL import ImageFont
# plt don't show 
mpl.use('Agg')
from datetime import datetime 
# from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier

def change_column_name_for_lgbm(column):
    if chr(34) in column:
        column = column.replace(chr(34), '')
    if chr(44) in column:
        column = column.replace(chr(44), '')
    if chr(58) in column:
        column = column.replace(chr(58), '')
    if chr(91) in column:
        column = column.replace(chr(91), '')
    if chr(93) in column:
        column = column.replace(chr(93), '')
    if chr(123) in column:
        column = column.replace(chr(123), '')
    if chr(125) in column:
        column = column.replace(chr(125), '')
        
    return column

def evaluation(y, pred, model_name='CatBoost', y_data_name='Test', verbose=False):
    # 2-3. 혼동 행렬 계산
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    # 2-4. 성능 평가 지표 계산
    accuracy = accuracy_score(y,pred)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred)  # Sensitivity (민감도)
    f1 = f1_score(y, pred)
    specificity = tn / (tn + fp)  # 특이도 계산
    mcc = matthews_corrcoef(y, pred)
    if verbose:
        # 2-5. 성능 평가 출력
        print(f"{model_name} {y_data_name} Accuracy      : {accuracy:.2f}")
        print(f"{model_name} {y_data_name} Precision     : {precision:.2f}")
        print(f"{model_name} {y_data_name} Recall (Sensitivity): {recall:.2f}")
        print(f"{model_name} {y_data_name} Specificity   : {specificity:.2f}")
        print(f"{model_name} {y_data_name} F1-Score      : {f1:.2f}")
        print(f"{model_name} {y_data_name} MCC           : {mcc:.2f}")
        print(f"{model_name} {y_data_name} Confusion Matrix TN / FP / FN / TP:  ({tn} / {fp} / {fn} / {tp})")
        print("\n\n-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
    
    return round(accuracy, 4), round(precision, 4), round(recall, 4), round(specificity, 4), round(f1, 4), round(tp, 4), round(fp, 4), round(fn, 4), round(tn, 4), round(mcc, 4)

# Feature Importance Plot (개별 모델)
def plot_feature_importance(model, features, title, save_path):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:30]  # 내림차순으로 정렬
    plt.figure(figsize=(20, 10))
    plt.barh(range(len(indices)), importance[indices], align="center", color="steelblue", edgecolor="black", alpha=0.8)
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=14)
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.ioff()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(tn, fp, fn, tp, model_name, save_path):
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    cm_percentage = cm_matrix / cm_matrix.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.matshow(cm_matrix, cmap="Blues", alpha=0.8)
    for (i, j), val in np.ndenumerate(cm_matrix):
        ax.text(j, i, f"{val}\n({cm_percentage[i, j]:.1f}%)", ha='center', va='center', 
                color="white" if cm_matrix[i, j] > cm_matrix.max() / 2 else "black")

    plt.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.ioff()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(save_path)
    plt.close()

    

def plot_ensemble_feature_importance(models, accuracies, features, title, save_path):
    importances = np.array([model.feature_importances_ for model in models])
    weights = np.array(accuracies)
    normalized_importance = []
    for i, imp in enumerate(importances):
        imp = (imp - imp.min()) / (imp.max() - imp.min())
        normalized_importance.append(imp)

    normalized_importance = np.array(normalized_importance)
    weights = np.array(accuracies)
    weighted_importances = (normalized_importance.T * weights).T

    combined_importance = weighted_importances.mean(axis=0)

    # combined_importance = (combined_importance - combined_importance.min()) / (combined_importance.max() - combined_importance.min())


    top_idx = np.argsort(combined_importance)[::-1][:30]
    features = np.array(features)
    top_features = [features[idx] for idx in top_idx]
    top_values = combined_importance[top_idx]
    np.set_printoptions(suppress=True, precision=4)
    plt.figure(figsize=(20, 10))
    plt.barh(range(len(top_values)), top_values, align="center", color="steelblue", edgecolor="black", alpha=0.8)
    plt.yticks(range(len(top_values)), top_features ,fontsize=14)
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.ioff()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_hist(prob, target, label, save_path):
    bins = np.arange(0, 1, 0.02)

    plt.figure(figsize=(20, 10))
    plt.hist(prob, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Probability", )
    plt.ylabel("Count")
    if label == "normal": 
        plt.title(f"{target} Histogram of normal probability")
    else:
        plt.title(f"{target} Histogram of abnormal probability")
    plt.xticks(np.arange(0, 1.2, 0.02), rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ioff()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance_shap(model, X, title, save_path, plot_type=None, tree=True):
    if tree == True:
        explainer = shap.TreeExplainer(model) 
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 500))
    shap_values = explainer.shap_values(X)  
    plt.figure(figsize=(8, 8))
    if plot_type == None:
        shap.summary_plot(shap_values, X, show=False, plot_size=(30,10), max_display=30)
    else:
        shap.summary_plot(shap_values, X, show=False, plot_size=(30,10), max_display=30, plot_type="bar")


    plt.title(title)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(save_path)
    plt.close()




