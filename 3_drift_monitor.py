import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

def check_data_drift():
    print("🔍 Avvio Sistema di Monitoraggio Drift (Kolmogorov-Smirnov Test)")
    df = pd.read_csv("data/raw/Variant II.csv")
    
    # Simuliamo il passaggio del tempo (Mesi 0-4 = Addestramento, Mesi 6-7 = Produzione/Nuovo Core Banking)
    baseline_data = df[df['month'] <= 4]
    current_data = df[df['month'] >= 6]
    
    features = ['income', 'credit_risk_score', 'proposed_credit_limit', 'customer_age']
    
    print("-" * 50)
    for feature in features:
        # Calcoliamo il test di Kolmogorov-Smirnov
        stat, p_value = ks_2samp(baseline_data[feature].dropna(), current_data[feature].dropna())
        
        if p_value < 0.01:
            print(f"⚠️  ALLARME DRIFT: '{feature}' ha cambiato distribuzione! (p-value: {p_value:.4e})")
        else:
            print(f"✅ STABILE: '{feature}' (Nessun drift rilevato)")
    print("-" * 50)
    print("Azione consigliata in caso di allarme: Triggerare riaddestramento automatico della pipeline.")

if __name__ == "__main__":
    check_data_drift()