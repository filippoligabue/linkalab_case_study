# SecureBank Fraud Detection System

## Struttura del Progetto
- `1_eda.py`: Script per l'Exploratory Data Analysis e generazione grafici (salvati in `/plots`).
- `2_pipeline.py`: Pipeline principale (Cross Validation, Preprocessing safe, Addestramento 3 modelli e calcolo "Business Recall").
- `3_drift_monitor.py`: Modulo di monitoraggio statistico post-migrazione core banking.

## Istruzioni di Esecuzione
1. Clonare la repository.
2. Posizionare il dataset in `data/raw/Variant II.csv`.
3. Installare le dipendenze: `pip install pandas numpy scikit-learn lightgbm imbalanced-learn matplotlib seaborn scipy`
4. Eseguire l'EDA: `python 1_eda.py`
5. Eseguire l'addestramento: `python 2_pipeline.py`
6. Testare il sistema di monitoraggio: `python 3_drift_monitor.py`

## Scelte Architetturali Chiave
- **Nessun Data Leakage:** Imputazione dei valori mancanti effettuata solo sui dati di addestramento.
- **Resilienza Core Banking:** Utilizzo di Ordinal Encoding con fallback per categorie ignote, evitando i crash tipici dell'One-Hot Encoding su sistemi dinamici.
- **Business Driven:** Modello ottimizzato su una metrica personalizzata calcolata al 99.75° percentile per rispettare il limite fisico di 500 calls/day.