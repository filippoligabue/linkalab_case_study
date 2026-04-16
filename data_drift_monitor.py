import pandas as pd
from scipy.stats import ks_2samp

def check_month7_drift(data_path):
    print("🔍 MONITORAGGIO DRIFT: Baseline (0-6) vs Test (Mese 7)")
    df = pd.read_csv(data_path)
    
    # Baseline: tutto tranne l'ultimo mese
    baseline = df[df['month'] < 7]
    # Target: l'ultimo mese (il più recente/migrazione)
    current = df[df['month'] == 7]
    
    features = ['income', 'credit_risk_score', 'proposed_credit_limit', 'customer_age', 'employment_status']
    
    print("-" * 60)
    print(f"{'Variabile':<25} | {'P-Value':<12} | {'Stato'}")
    print("-" * 60)
    
    for feat in features:
        if df[feat].dtype in ['float64', 'int64']:
            stat, p_val = ks_2samp(baseline[feat].dropna(), current[feat].dropna())
            status = "⚠️ DRIFT" if p_val < 0.01 else "✅ STABILE"
            print(f"{feat:<25} | {p_val:<12.4e} | {status}")
        else:
            # Per le categoriche calcoliamo la variazione delle frequenze
            diff = (baseline[feat].value_counts(normalize=True) - 
                    current[feat].value_counts(normalize=True)).abs().sum()
            status = "⚠️ DRIFT" if diff > 0.1 else "✅ STABILE"
            print(f"{feat:<25} | {diff:<12.4f} | {status} (Freq Delta)")

if __name__ == "__main__":
    check_month7_drift("data/raw/Variant II.csv")