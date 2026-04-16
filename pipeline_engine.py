import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

def clean_and_feature_engineering(df):
    print("--- 🛠️ Preprocessing & Feature Engineering ---")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    zero_var_num = [col for col in num_cols if df[col].std() == 0]
    zero_var_cat = [col for col in cat_cols if df[col].nunique() <= 1]
    
    cols_to_drop = zero_var_num + zero_var_cat
    df = df.drop(columns=cols_to_drop)
    print(f"   > Rimosse {len(cols_to_drop)} variabili con varianza zero: {cols_to_drop}")
    
    df['limit_to_income'] = df['proposed_credit_limit'] / (df['income'] + 0.01)
    df['age_group'] = pd.cut(df['customer_age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    
    num_cols_remaining = df.select_dtypes(include=['float64', 'int64']).columns.drop(['fraud_bool', 'month'])
    scaler = MinMaxScaler()
    df[num_cols_remaining] = scaler.fit_transform(df[num_cols_remaining])
    
    cat_cols_remaining = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols_remaining:
        df[col] = df[col].astype(str).replace('nan', 'UNKNOWN')
        
    return df

def evaluate_business_matrix(y_true, y_prob, name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Vincolo Operativo: 500 chiamate / 200.000 transazioni = 0.25% FPR
    idx = np.where(fpr <= 0.0025)[0][-1]
    threshold = thresholds[idx]
    
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    total = len(y_true)
    tp_total_perc = (cm[1,1] / total) * 100
    recall = (cm[1,1] / (cm[1,1] + cm[1,0])) * 100
    
    print(f"\n📊 BUSINESS ANALYSIS: {name}")
    print(f"Soglia (FPR target 0.25%): {threshold:.4f}")
    print(f"--------------------------------------------------")
    print(f"TN: {cm[0,0]:<8} | FP: {cm[0,1]:<8} (Chiamate effettuate)")
    print(f"FN: {cm[1,0]:<8} | TP: {cm[1,1]:<8} (Frodi bloccate)")
    print(f"--------------------------------------------------")
    print(f"Recall: {recall:.2f}%")
    print(f"Efficienza (TP su Totale): {tp_total_perc:.4f}%")
    
    benchmark = 0.12 # 40% di recall su 0.3% di prevalenza
    if tp_total_perc > benchmark:
        print(f"🚀 VITTORIA! Superato il sistema a regole ({benchmark}%).")
    else:
        print(f"⚠️  SOTTO BENCHMARK. Il sistema a regole è ancora superiore.")
    
    return tp_total_perc

def run_v5():
    # Caricamento
    df = pd.read_csv("data/raw/Variant II.csv") 
    df = clean_and_feature_engineering(df)
    
    train_df = df[df['month'] <= 5]
    test_df = df[df['month'] >= 6]
    
    X_train = train_df.drop(columns=['fraud_bool', 'month'])
    y_train = train_df['fraud_bool']
    X_test = test_df.drop(columns=['fraud_bool', 'month'])
    y_test = test_df['fraud_bool']
    
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- 1. CATBOOST - Cost-Sensitive ---
    print("\n[1/3] Training CatBoost (Weighted)...")
    model_cat = CatBoostClassifier(iterations=1000, auto_class_weights='Balanced', 
                                   cat_features=cat_features, verbose=0)
    model_cat.fit(X_train, y_train)
    evaluate_business_matrix(y_test, model_cat.predict_proba(X_test)[:, 1], "CatBoost CS")

    # Encoder condiviso per LightGBM
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr_enc = enc.fit_transform(X_train)
    X_ts_enc = enc.transform(X_test)
    
    # --- 2. LIGHTGBM - SMOTE + Under (Ibrido) ---
    print("\n[2/3] Training LightGBM (Hybrid Resampling SMOTE+Under)...")
    pipe_hybrid = Pipeline([
        ('smote', SMOTE(sampling_strategy=0.05, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=0.2, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42))
    ])
    pipe_hybrid.fit(X_tr_enc, y_train)
    evaluate_business_matrix(y_test, pipe_hybrid.predict_proba(X_ts_enc)[:, 1], "LGBM SMOTE+Under")
    
    # --- 3. LIGHTGBM - Random UnderSampler Puro ---
    print("\n[3/3] Training LightGBM (Random Undersampling)...")
    pipe_under = Pipeline([
        ('under', RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42))
    ])
    pipe_under.fit(X_tr_enc, y_train)
    evaluate_business_matrix(y_test, pipe_under.predict_proba(X_ts_enc)[:, 1], "LGBM UnderSampling")

    # --- Feature Importance Output (Modello Vincente: CatBoost) ---
    print("\nGenerazione plot Feature Importance per CatBoost...")
    os.makedirs("plots/models", exist_ok=True)
    importances = model_cat.get_feature_importance()
    feat_imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), palette='mako')
    plt.title('Top 15 Feature Importance - CatBoost')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('plots/models/catboost_feature_importance.png')
    plt.close()
    print("Plot salvato con successo in 'plots/models/catboost_feature_importance.png' 🎯")

if __name__ == "__main__":
    run_v5()