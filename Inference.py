from utils import *  
import os
import re
import json
import joblib
import pickle
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score



def clean_column_name(column: str) -> str:
    return re.sub(r'[",:<>{}\[\]]', "_", column)


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns={c: clean_column_name(c) for c in df.columns}, inplace=True)
    return df


def load_scaler_or_fit(path: str, data: pd.DataFrame, cols: list):
    if os.path.exists(path):
        scaler = pickle.load(open(path, "rb"))
        print(f"✅ Loaded existing scaler from {path}")
    else:
        scaler = StandardScaler()
        scaler.fit(data[cols])
        pickle.dump(scaler, open(path, "wb"))
        print(f"💾 Saved new scaler to {path}")
    return scaler


def run_inference():
    base_dir = "/workspace/source/code_je/251104"
    data_path = f"{base_dir}/clean_data/Prep_251104.csv"
    json_path = f"{base_dir}/json/features.json"
    scaler_dir = f"{base_dir}/scaler"
    model_root = "/workspace/source/test/20251014/Results/Rate_3/Weights"
    today = datetime.today().strftime('%Y%m%d')

    df = load_and_clean_data(data_path)

    # 타겟 질환별 반복
    for target_name in ["뇌졸중", "심장병(심근경색및협심증)"]:
        print("\n" + "="*80)
        print(f"Running inference for [{target_name}]")
        print("="*80)

        with open(json_path, "r") as f:
            feat_json = json.load(f)
        features = [clean_column_name(c) for c in feat_json[target_name]["features"]]
        label_col = feat_json[target_name]["labels"]

        scaler_path = os.path.join(scaler_dir, f"z_score_{target_name}.pkl")
        scaler = load_scaler_or_fit(scaler_path, df, features)
        df[features] = scaler.transform(df[features])

        if target_name == "뇌졸중":
            model_subdir = "20251014_Without_10%_Features_Change_Normal"
        else:
            model_subdir = "20251014_Duplicates_Without_15%_Features_ver2"

        cat_path = os.path.join(model_root, model_subdir, f"cat_{target_name}.pkl")
        xgb_path = os.path.join(model_root, model_subdir, f"xgb_{target_name}.pkl")
        lgb_path = os.path.join(model_root, model_subdir, f"lgbm_{target_name}.pkl")

        cat_model = joblib.load(cat_path)
        xgb_model = joblib.load(xgb_path)
        lgbm_model = joblib.load(lgb_path)

        X = df[features]
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_feats = set(cat_model.feature_names_)
        test_feats = set(X.columns)
        if train_feats != test_feats:
            missing = train_feats - test_feats
            extra = test_feats - train_feats
            raise ValueError(f"Feature mismatch detected!\nMissing: {missing}\nExtra: {extra}")

        cat_pred = cat_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
        lgbm_pred = lgbm_model.predict_proba(X_test)[:, 1]

        ensemble_pred = (cat_pred + lgbm_pred + xgb_pred) / 3

        auc = roc_auc_score(y_test, ensemble_pred)
        print(f"{target_name} Soft Voting ROC-AUC: {auc:.4f}")


        save_dir = f"/workspace/source/test/{today}/Results/Inference_{target_name}"
        os.makedirs(save_dir, exist_ok=True)
        result_df = pd.DataFrame({
            "S_PID": df.loc[X_test.index, "S_PID"],
            "Pred_Ensemble": ensemble_pred,
            "True_Label": y_test
        })
        result_df.to_csv(f"{save_dir}/inference_result.csv", index=False)
        print(f"Saved results → {save_dir}/inference_result.csv")



if __name__ == "__main__":
    run_inference()
