# 🚀 SecureBank S.p.A. - Fraud Detection Engine

Questo repository contiene la soluzione di Machine Learning sviluppata per il rilevamento di transazioni bancarie fraudolente per conto di SecureBank S.p.A. 

Il progetto affronta la sfida dell'**estremo sbilanciamento delle classi** (solo lo 0.3% di transazioni fraudolente) e integra logiche stringenti di business, ottimizzando il modello per massimizzare la *detection rate* senza superare la capacità operativa del call center aziendale.

## 📊 Contesto di Business e Vincoli
Il sistema basato su regole preesistente individuava solo il 40% delle frodi (benchmark di efficienza: **0.12%** sul totale delle transazioni). 
Il vincolo principale del nuovo sistema ML è dettato dalla capacità del call center: **massimo 500 chiamate di verifica al giorno su un volume di 200.000 transazioni**.
Tradotto in metriche ML, il sistema è vincolato a operare con un **False Positive Rate (FPR) massimo dello 0.25%**. Tutte le soglie decisionali dei modelli sono state ricalibrate dinamicamente per rispettare questa precisa percentuale.

## 📂 Struttura del Progetto

* `exploratory_data_analysis.py`: Script per la generazione di grafici esplorativi (sbilanciamento, bias anagrafico, distribuzioni numeriche, correlazioni e feature importance tramite Mutual Information).
* `fraud_detection_training.py`: La pipeline core. Esegue feature engineering, normalizzazione e addestra tre diversi approcci per lo sbilanciamento (CatBoost Cost-Sensitive, LightGBM con SMOTE+UnderSampling, LightGBM con UnderSampling puro).
* `data_drift_monitor.py`: Modulo statistico per rilevare scostamenti nelle distribuzioni dei dati in vista della migrazione del nuovo sistema di Core Banking.
* `test_pipeline.py`: Suite di Unit Test basata su `pytest` per garantire la stabilità e la correttezza matematica della fase di preprocessing.
* `Dockerfile` & `docker-compose.yml`: Infrastruttura per eseguire il servizio di monitoraggio del drift in un ambiente isolato.
* `requirements.txt`: Elenco delle dipendenze Python necessarie.

## ⚙️ Installazione e Setup

1. Clonare la repository in locale.
2. Creare un virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Su Windows: env\Scripts\activate