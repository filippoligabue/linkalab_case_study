import pytest
import pandas as pd
import numpy as np
from fraud_detection_training import clean_and_feature_engineering

@pytest.fixture
def sample_dataframe():
    """
    Fixture che crea un dataset fittizio con casi limite per testare la pipeline di preprocessing.
    """
    return pd.DataFrame({
        'fraud_bool': [0, 1, 0, 0, 1],
        'month': [1, 1, 2, 2, 3],
        'income': [1000, 2000, 1500, 3000, 2500],
        'proposed_credit_limit': [500, 1000, 500, 1500, 1000],
        'customer_age': [25, 35, 50, 70, 40],
        'device_fraud_count': [0, 0, 0, 0, 0], # Colonna a varianza ZERO (da rimuovere)
        'categorical_col': ['A', 'B', np.nan, 'A', 'C'] # Presenza di NaN
    })

def test_clean_and_feature_engineering_removes_zero_variance(sample_dataframe):
    """Testa che le feature con varianza zero vengano eliminate."""
    processed_df = clean_and_feature_engineering(sample_dataframe.copy())
    
    assert 'device_fraud_count' not in processed_df.columns, "La colonna a varianza zero non è stata rimossa!"

def test_clean_and_feature_engineering_creates_features(sample_dataframe):
    """Testa la creazione delle feature derivate e verifica che subiscano lo scaling."""
    processed_df = clean_and_feature_engineering(sample_dataframe.copy())
    
    # Verifichiamo che le colonne siano state effettivamente create
    assert 'limit_to_income' in processed_df.columns, "Feature limit_to_income mancante"
    assert 'age_group' in processed_df.columns, "Feature age_group mancante"
    
    # Siccome la pipeline applica il MinMaxScaler, i valori finali 
    # della colonna 'limit_to_income' devono essere compressi tra 0 e 1.
    assert processed_df['limit_to_income'].min() >= 0.0, "Il valore minimo dopo lo scaling dovrebbe essere >= 0"
    assert processed_df['limit_to_income'].max() <= 1.0, "Il valore massimo dopo lo scaling dovrebbe essere <= 1"

def test_clean_and_feature_engineering_handles_nans(sample_dataframe):
    """Testa la sostituzione corretta dei NaN nelle colonne categoriche."""
    processed_df = clean_and_feature_engineering(sample_dataframe.copy())
    
    # Verifica che la parola 'nan' o il vero NaN sia diventato 'UNKNOWN' (o la logica definita)
    assert processed_df['categorical_col'].isna().sum() == 0, "Ci sono ancora veri NaN nel dataframe"
    assert processed_df['categorical_col'].iloc[2] == 'UNKNOWN', "I NaN non sono stati rimpiazzati con 'UNKNOWN'"

def test_scaling_preserves_target_and_month(sample_dataframe):
    """Testa che il target e la variabile temporale non vengano alterati dallo scaler."""
    processed_df = clean_and_feature_engineering(sample_dataframe.copy())
    
    # Valori originali vs scalati
    assert processed_df['fraud_bool'].tolist() == [0, 1, 0, 0, 1]
    assert processed_df['month'].tolist() == [1, 1, 2, 2, 3]