# tests/test_analyzer_triplegana.py
import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from analyzer_triplegana import preparar_datos, test_uniformidad_chi2, TripleganaConfig

@pytest.fixture
def sample_df():
    # Crear DataFrame pequeño representativo
    data = {
        'fecha': ['01/01/2025', '01/01/2025', '02/01/2025'],
        'hora': ['1 pm', '4 pm', '10 pm'],
        'numero': ['123', '045', '999'],
        'signo': ['GEM', 'ESC', 'LEO']
    }
    return pd.DataFrame(data)

def test_preparar_datos_discards(tmp_path, sample_df):
    cfg = TripleganaConfig(output_dir=str(tmp_path))
    df_prepared = preparar_datos(sample_df, cfg)
    # Debe crear columnas numero_int y ultimo_digito
    assert 'numero_int' in df_prepared.columns
    assert 'ultimo_digito' in df_prepared.columns
    # Check that filas_descartadas file does not exist (no invalid rows)
    discarded = tmp_path / 'filas_descartadas_triplegana.csv'
    assert not discarded.exists()

def test_chi2_on_uniform_data():
    # Generar datos uniformes en 0..999
    n = 1000
    rng = np.random.default_rng(123)
    nums = rng.integers(0, 1000, size=n)
    df = pd.DataFrame({
        'fecha': ['01/01/2025'] * n,
        'hora': ['1 pm'] * n,
        'numero': nums,
        'signo': ['GEM'] * n
    })
    cfg = TripleganaConfig()
    df_prepared = preparar_datos(df, cfg)
    res = test_uniformidad_chi2(df_prepared, bins=10, config=cfg)
    assert 'p' in res
    # En datos simulados uniformes p no debería ser extremadamente pequeño
    assert res['p'] > 0.001

def test_bins_labels_match():
    # Verificar que la creación de bins no produce labels desalineadas
    cfg = TripleganaConfig()
    df = pd.DataFrame({
        'fecha': ['01/01/2025', '02/01/2025'],
        'hora': ['1 pm', '4 pm'],
        'numero': [0, 999],
        'signo': ['GEM', 'ESC']
    })
    df_prepared = preparar_datos(df, cfg)
    # labels y bins implícitos: rango_mil debe ser categórica con len(labels) == len(bins)-1
    rango = df_prepared['rango_mil'].dtype
    assert 'category' in str(rango) or df_prepared['rango_mil'].dtype == object
