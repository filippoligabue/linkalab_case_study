import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import os
import warnings

warnings.filterwarnings('ignore')

def run_full_eda(data_path):
    # Creazione cartella dedicata
    out_dir = "plots/eda"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Caricamento dati per EDA Avanzata...")
    df = pd.read_csv(data_path)
    
    # 1. Distribuzione Target
    print("Generazione grafico sbilanciamento classi...")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='fraud_bool', data=df, palette='Set2')
    plt.title("Distribuzione delle Frodi (Scala Logaritmica)")
    plt.yscale('log')
    plt.savefig(f"{out_dir}/1_target_distribution.png", bbox_inches='tight')
    plt.close()

    # 2. Analisi Bias Età (Fondamentale per Variant II)
    print("Generazione grafico Bias Età (Variant II)...")
    age_fraud_rate = df.groupby('customer_age')['fraud_bool'].mean() * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(x=age_fraud_rate.index, y=age_fraud_rate.values, palette="viridis")
    plt.title("Tasso di frode % per età (Analisi Bias Variant II)")
    plt.ylabel("% di Frodi")
    plt.xlabel("Età del cliente")
    plt.savefig(f"{out_dir}/2_age_bias.png", bbox_inches='tight')
    plt.close()

    # 3. Analisi Valori Mancanti (Prendendo spunto dal notebook)
    # Nel dataset, i valori < 0 indicano spesso missing values
    print("Generazione analisi valori mancanti...")
    missing_cols = ['prev_address_months_count', 'current_address_months_count', 'bank_months_count']
    missing_data = []
    for col in missing_cols:
        if col in df.columns:
            missing_perc = (df[col] < 0).mean() * 100
            missing_data.append({'Feature': col, 'Missing %': missing_perc})
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Missing %', y='Feature', data=missing_df, palette='Reds_r')
        plt.title("Percentuale di Valori Mancanti (Valori < 0)")
        plt.savefig(f"{out_dir}/3_missing_values.png", bbox_inches='tight')
        plt.close()

    # 4. Distribuzione Feature Numeriche per Classe (KDE Plots)
    print("Generazione distribuzioni features chiave...")
    # Selezioniamo alcune numeriche interessanti per non intaccare la memoria
    num_features = ['income', 'customer_age', 'credit_risk_score', 'proposed_credit_limit']
    num_features = [f for f in num_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, feature in enumerate(num_features):
        sns.kdeplot(data=df, x=feature, hue='fraud_bool', common_norm=False, 
                    fill=True, alpha=0.3, ax=axes[i], palette='Set1')
        axes[i].set_title(f'Distribuzione di {feature}')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_numeric_distributions.png", bbox_inches='tight')
    plt.close()

    # 5. Matrice di Correlazione (Pearson)
    print("Generazione Matrice di Correlazione...")
    # Seleziono solo le numeriche per la correlazione lineare
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Maschera per il triangolo superiore
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                vmax=1, vmin=-1, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Matrice di Correlazione (Pearson)")
    plt.savefig(f"{out_dir}/5_correlation_matrix.png", bbox_inches='tight')
    plt.close()

    # 6. Feature Significance (Mutual Information) - Sostituisce ExtraTrees per velocità
    print("Calcolo Feature Importance (Mutual Information)...")
    # Facciamo un campionamento per non bloccare il computer (calcolare MI su 1M righe è lento)
    df_sample = df.sample(n=50000, random_state=42)
    X_sample = df_sample.drop(columns=['fraud_bool'])
    y_sample = df_sample['fraud_bool']
    
    # Riempiamo temporaneamente i nan/valori negativi per il calcolo
    for col in X_sample.select_dtypes(include=[np.number]).columns:
        X_sample[col] = np.where(X_sample[col] < 0, np.nan, X_sample[col])
        X_sample[col] = X_sample[col].fillna(X_sample[col].median())
    
    # Codifichiamo brutalmente le categoriche solo per questo test
    cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X_sample[col] = X_sample[col].astype('category').cat.codes
        
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
    mi_scores_series = pd.Series(mi_scores, index=X_sample.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=mi_scores_series.values[:15], y=mi_scores_series.index[:15], palette='mako')
    plt.title("Top 15 Features più significative (Mutual Information)")
    plt.xlabel("Mutual Information Score")
    plt.savefig(f"{out_dir}/6_feature_importance.png", bbox_inches='tight')
    plt.close()

    print(f"EDA completata con successo! Tutti i grafici sono in '{out_dir}/'")

if __name__ == "__main__":
    # Assicurati che il path sia corretto
    run_full_eda("data/raw/Variant II.csv")