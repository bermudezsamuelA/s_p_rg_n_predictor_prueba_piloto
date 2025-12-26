# tests/test_analyzer_supergana.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from analyzer_supergana import preparar_datos, test_uniformidad_chi2, AnalyzerConfig

@pytest.fixture
def sample_df():
    data = {
        'fecha': ['01/01/2025', '01/01/2025', '02/01/2025'],
        'hora': ['1 pm', '4 pm', '10 pm'],
        'numero': ['1234', '0456', '999'],
        'signo': ['GEM', 'ESC', 'LEO']
    }
    return pd.DataFrame(data)

def test_preparar_datos_basic(tmp_path, sample_df):
    cfg = AnalyzerConfig(output_dir=str(tmp_path))
    df_prepared = preparar_datos(sample_df, cfg)
    assert 'numero_int' in df_prepared.columns
    assert 'ultimo_digito' in df_prepared.columns
    # filas descartadas no deben existir en este caso
    discarded = Path(cfg.output_dir) / 'filas_descartadas_supergana.csv'
    assert not discarded.exists()

def test_chi2_on_uniform_data():
    n = 1000
    rng = np.random.default_rng(123)
    nums = rng.integers(0, 10000, size=n)
    df = pd.DataFrame({
        'fecha': ['01/01/2025'] * n,
        'hora': ['1 pm'] * n,
        'numero': nums,
        'signo': ['GEM'] * n
    })
    cfg = AnalyzerConfig()
    df_prepared = preparar_datos(df, cfg)
    res = test_uniformidad_chi2(df_prepared, bins=10, config=cfg)
    assert 'p' in res
    assert res['p'] > 0.001  # no debería rechazar uniformidad con alta probabilidad

def test_bins_labels_match():
    cfg = AnalyzerConfig()
    df = pd.DataFrame({
        'fecha': ['01/01/2025', '02/01/2025'],
        'hora': ['1 pm', '4 pm'],
        'numero': [0, 9999],
        'signo': ['GEM', 'ESC']
    })
    df_prepared = preparar_datos(df, cfg)
    # rango_mil debe existir y ser categórica o string
    assert 'rango_mil' in df_prepared.columns
