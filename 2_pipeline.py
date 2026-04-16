import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings('ignore')

# Costanti
DATA_PATH = "data/raw/Variant II.csv" # Modifica con il tuo path
FPR_LIMIT = 0.0025  # 500 chiamate su ~200k transazioni

def evaluate_business_recall(y_true, y_pred_proba, fpr_limit):
    """Calcola la Recall massima mantenendo l'FPR sotto il limite stabilito."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # Trova l'indice dell'ultimo FPR che è <= al nostro limite
    valid_indices = np.where(fpr <= fpr_limit)[0]
    if len(valid_indices) == 0:
        return 0.0, 1.0
    idx = valid_indices[-1]
    return tpr[idx] * 100, thresholds[idx]

def run_pipeline():
    print("🚀 Inizio Pipeline SecureBank - Rilevamento Frodi\n")
    df = pd.read_csv(DATA_PATH)
    
    # Prepariamo X e y (Teniamo TUTTE le variabili come hai giustamente intuito)
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']
    
    # Identifichiamo colonne numeriche e categoriche
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    results = {'Cost-Sensitive': [], 'Undersampling': [], 'SMOTE': []}
    
    print("🔄 Esecuzione Time-Series Cross Validation (3 Folds)...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n   ➤ Fold {fold + 1}/3")
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # --- 1. GESTIONE MISSING VALUES (NO LEAKAGE) ---
        # Imputiamo con la mediana calcolata SOLO sul train
        for col in ['prev_address_months_count', 'current_address_months_count', 'bank_months_count']:
            if col in X_train.columns:
                X_train.loc[X_train[col] < 0, col] = np.nan
                X_test.loc[X_test[col] < 0, col] = np.nan
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
                
        # --- 2. ENCODING CATEGORICO SICURO ---
        # OrdinalEncoder gestisce nuove categorie ignorandole (non esplode come get_dummies)
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cat_cols:
            X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = encoder.transform(X_test[cat_cols])

        # --- APPROCCIO 1: Cost-Sensitive Learning ---
        # Diamo un peso enorme alla classe minoritaria
        clf_cs = LGBMClassifier(scale_pos_weight=100, n_estimators=200, random_state=42, n_jobs=-1)
        clf_cs.fit(X_train, y_train)
        preds_cs = clf_cs.predict_proba(X_test)[:, 1]
        rec_cs, _ = evaluate_business_recall(y_test, preds_cs, FPR_LIMIT)
        results['Cost-Sensitive'].append(rec_cs)

        # --- APPROCCIO 2: Undersampling ---
        rus = RandomUnderSampler(random_state=42)
        X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
        clf_rus = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf_rus.fit(X_train_rus, y_train_rus)
        preds_rus = clf_rus.predict_proba(X_test)[:, 1]
        rec_rus, _ = evaluate_business_recall(y_test, preds_rus, FPR_LIMIT)
        results['Undersampling'].append(rec_rus)

        # --- APPROCCIO 3: SMOTE ---
        # Riempiamo i rimanenti NaN per SMOTE (SMOTE non accetta NaN)
        X_train_sm = X_train.fillna(-999) 
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_sm, y_train)
        clf_smote = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf_smote.fit(X_train_smote, y_train_smote)
        preds_smote = clf_smote.predict_proba(X_test.fillna(-999))[:, 1]
        rec_smote, _ = evaluate_business_recall(y_test, preds_smote, FPR_LIMIT)
        results['SMOTE'].append(rec_smote)

    print("\n" + "="*50)
    print("🎯 RISULTATI FINALI (Media su 3 Fold Temporali)")
    print(f"Obiettivo: Massimizzare la Recall con Max 500 Falsi Positivi/Giorno (FPR = {FPR_LIMIT*100}%)")
    print("="*50)
    for name, metrics in results.items():
        print(f" - {name:15}: Business Recall Media = {np.mean(metrics):.2f}%")

if __name__ == "__main__":
    run_pipeline()