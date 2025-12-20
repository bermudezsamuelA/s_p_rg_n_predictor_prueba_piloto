"""
analyzer_triplegana.py — Analyzer optimizado para Triple Gana
Mejoras: CONFIG tipado, logging, headless, exportación de resúmenes,
validaciones estadísticas, visualizaciones y dataset ML.
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Optional statsmodels
try:
    from statsmodels.tsa.stattools import acf as sm_acf
    from statsmodels.sandbox.stats.runs import runstest_1samp
except Exception:
    sm_acf = None
    runstest_1samp = None

# -------------------------
# CONFIG
# -------------------------
@dataclass
class TripleganaConfig:
    db_path: str = 'data/triplegana.db'
    table_name: str = 'triplegana_resultados'
    output_dir: str = 'data/analysis_triplegana'
    figures_dir: str = 'data/analysis_triplegana/figures'
    headless: bool = True
    shapiro_max: int = 5000
    chi_bins: int = 20
    rolling_windows: Tuple[int, int] = (7, 30)
    uniformity_alpha: float = 0.05
    random_state: int = 42
    ml_target: str = 'ultimo_digito'
    export_summaries: bool = True
    display_top_n: int = 20
    log_level: str = 'INFO'
    feature_columns: Tuple = ('hora_num', 'dia_semana', 'mes', 'suma_digitos', 'paridad', 'signo_cod', 'dias_desde_inicio')

CONFIG = TripleganaConfig()
Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
Path(CONFIG.figures_dir).mkdir(parents=True, exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=getattr(logging, CONFIG.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(CONFIG.output_dir) / 'analyzer_triplegana.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# UTILIDADES
# -------------------------
def _save_json(obj: Any, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)

def _safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# -------------------------
# CARGA Y VALIDACIÓN
# -------------------------
def cargar_datos(config: TripleganaConfig = CONFIG) -> Optional[pd.DataFrame]:
    logger.info("Cargando datos desde %s", config.db_path)
    if not Path(config.db_path).exists():
        logger.error("Archivo no encontrado: %s", config.db_path)
        return None
    try:
        with sqlite3.connect(config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (config.table_name,))
            if not cursor.fetchone():
                logger.error("Tabla %s no encontrada en %s", config.table_name, config.db_path)
                return None
            df = pd.read_sql_query(f"SELECT * FROM {config.table_name} ORDER BY fecha, hora", conn)
            logger.info("Registros cargados: %d", len(df))
            return df
    except Exception as e:
        logger.exception("Error cargando datos: %s", e)
        return None

# -------------------------
# PREPARACIÓN DE DATOS
# -------------------------
def preparar_datos(df: pd.DataFrame, config: TripleganaConfig = CONFIG) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        logger.warning("DataFrame vacío en preparar_datos")
        return None

    df = df.copy()

    # Fecha
    df['fecha_raw'] = df.get('fecha')
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')

    # Hora vectorizada (maneja 12am/12pm)
    s = df['hora'].astype(str).str.lower().str.strip()
    is_pm = s.str.contains('pm', na=False)
    is_am = s.str.contains('am', na=False)
    horas = s.str.extract(r'(\d+)')[0]
    horas = pd.to_numeric(horas, errors='coerce')
    horas = horas.fillna(-1).astype(float)
    # aplicar transformaciones vectorizadas con cuidado de NaNs
    mask_pm = is_pm & (horas != 12)
    mask_am_12 = is_am & (horas == 12)
    horas.loc[mask_pm] = horas.loc[mask_pm] + 12
    horas.loc[mask_am_12] = 0
    df['hora_num'] = horas.fillna(-1).astype(int)

    # Número y validaciones
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
    before = len(df)

    # Exportar filas descartadas por columnas críticas antes de dropear
    invalid_mask = df[['fecha', 'numero', 'hora', 'signo']].isna().any(axis=1)
    if invalid_mask.any():
        discarded = df[invalid_mask].copy()
        discarded_path = Path(config.output_dir) / 'filas_descartadas_triplegana.csv'
        discarded.to_csv(discarded_path, index=False, encoding='utf-8')
        logger.info("Filas descartadas exportadas a %s (count=%d)", discarded_path, len(discarded))

    df = df.dropna(subset=['fecha', 'numero', 'hora', 'signo'])
    dropped = before - len(df)
    if dropped:
        logger.info("Filas descartadas por datos inválidos: %d", dropped)

    # Asegurar entero
    df['numero'] = df['numero'].astype(int)

    # Features
    df['ultimo_digito'] = df['numero'] % 10
    df['penultimo_digito'] = (df['numero'] // 10) % 10
    df['suma_digitos'] = df['numero'].astype(str).apply(lambda s: sum(int(c) for c in s))
    df['paridad'] = df['numero'] % 2
    df['signo'] = df['signo'].astype(str)
    df['signo_cod'] = pd.factorize(df['signo'])[0].astype('int8')

    # Validación antes de crear rangos
    if df['numero'].isna().any():
        logger.warning("Existen valores NaN en 'numero' antes de crear rangos; serán descartados o imputados.")

    # Rangos por mil (robusto, dinámico según datos)
    try:
        min_num = int(df['numero'].min())
        max_num = int(df['numero'].max())
        # Redondear límites a múltiplos de 1000
        min_bin = (min_num // 1000) * 1000
        top = ((max_num // 1000) + 1) * 1000
        # Evitar bins vacíos si min==max
        if top <= min_bin:
            top = min_bin + 1000
        bins_fijos = list(range(min_bin, top + 1, 1000))
        labels_fijos = [f'{bins_fijos[i]}-{bins_fijos[i+1]-1}' for i in range(len(bins_fijos)-1)]

        logger.debug("bins_fijos: %s", bins_fijos)
        logger.debug("labels_fijos: %s", labels_fijos)

        df['rango_mil'] = pd.cut(df['numero'],
                                 bins=bins_fijos,
                                 labels=labels_fijos,
                                 right=False,
                                 include_lowest=True)

        # Marcar valores fuera de rango (NaN) con etiqueta 'out_of_range'
        df['rango_mil'] = df['rango_mil'].cat.add_categories(['out_of_range']).fillna('out_of_range')
    except Exception as e:
        logger.exception("Error creando rango_mil dinámico: %s", e)
        df['rango_mil'] = 'out_of_range'

    # Temporales
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_mes'] = df['fecha'].dt.day
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dias_desde_inicio'] = (df['fecha'] - df['fecha'].min()).dt.days.astype(int)

    # Guardar preparado
    out_path = Path(config.output_dir) / 'triplegana_preparado.csv'
    df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info("Datos preparados guardados en %s", out_path)
    return df

# -------------------------
# ANÁLISIS Y TESTS
# -------------------------
def test_uniformidad_chi2(df: pd.DataFrame, bins: int, config: TripleganaConfig = CONFIG) -> Dict[str, Any]:
    n = len(df)
    try:
        freq_obs, edges = np.histogram(df['numero'], bins=bins)
        expected = np.array([n / bins] * bins)
        if (expected < 5).any():
            logger.warning("Algunas celdas esperadas < 5; chi-square puede no ser válido")
        chi2, p = stats.chisquare(freq_obs, expected)
        return {'chi2': float(chi2), 'p': float(p), 'uniforme': p > config.uniformity_alpha}
    except Exception as e:
        logger.exception("Error en chi2: %s", e)
        return {'error': str(e)}

def analisis_completo(df: pd.DataFrame, config: TripleganaConfig = CONFIG) -> Dict[str, Any]:
    logger.info("Iniciando análisis completo Triple Gana")
    out = {}
    out['n'] = len(df)
    out['basic_stats'] = {
        'mean': float(df['numero'].mean()),
        'median': float(df['numero'].median()),
        'std': float(df['numero'].std())
    }

    # Último dígito
    ult = df['ultimo_digito'].value_counts().sort_index()
    out['ultimo_digito'] = {
        'frecuencias': ult.to_dict(),
        'porcentajes': (ult / len(df) * 100).round(3).to_dict()
    }
    esperado = len(df) / 10
    desviacion = ((ult - esperado) / esperado * 100).round(2)
    out['ultimo_digito']['calientes'] = desviacion[desviacion > 10].index.tolist()
    out['ultimo_digito']['frios'] = desviacion[desviacion < -10].index.tolist()

    # Paridad
    out['paridad'] = df['paridad'].value_counts().to_dict()

    # Signos
    out['signos'] = df['signo'].value_counts().to_dict()

    # Correlaciones
    numeric_cols = ['numero', 'hora_num', 'dia_semana', 'mes', 'ultimo_digito', 'suma_digitos', 'dias_desde_inicio']
    corr = df[numeric_cols].corr()
    out['corr_with_numero'] = corr['numero'].sort_values(ascending=False).to_dict()

    # Uniformidad
    out['chi2_uniform'] = test_uniformidad_chi2(df, config.chi_bins, config)

    # Kruskal por hora
    try:
        groups = [g['numero'].values for _, g in df.groupby('hora_num')]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            out['kruskal_hora'] = {'h': float(h), 'p': float(p), 'significativo': p < config.uniformity_alpha}
    except Exception as e:
        logger.warning("Kruskal por hora falló: %s", e)

    # Autocorrelación
    if sm_acf is not None:
        try:
            acf_vals = sm_acf(df['numero'].values, nlags=20, fft=False)
            out['acf'] = [float(x) for x in acf_vals]
        except Exception as e:
            logger.debug("ACF falló: %s", e)
            out['acf'] = None
    else:
        out['acf'] = None

    # Runs test
    if runstest_1samp is not None:
        try:
            stat, p = runstest_1samp(df['numero'])
            out['runs_test'] = {'stat': float(stat), 'p': float(p), 'aleatorio': p > config.uniformity_alpha}
        except Exception as e:
            logger.debug("runstest_1samp falló: %s", e)
            out['runs_test'] = None
    else:
        out['runs_test'] = None

    # Exportar resúmenes
    if config.export_summaries:
        _safe_mkdir(Path(config.output_dir))
        pd.Series(out['ultimo_digito']['frecuencias']).to_csv(Path(config.output_dir) / 'triplegana_ultimo_digito.csv')
        pd.Series(out['paridad']).to_csv(Path(config.output_dir) / 'triplegana_paridad.csv')
        pd.Series(out['signos']).to_csv(Path(config.output_dir) / 'triplegana_signos.csv')
        corr.to_csv(Path(config.output_dir) / 'triplegana_corr.csv')
        _save_json(out, Path(config.output_dir) / 'triplegana_tests.json')
        logger.info("Resúmenes exportados a %s", config.output_dir)

    return out

# -------------------------
# VISUALIZACIONES
# -------------------------
def visualizar(df: pd.DataFrame, analysis: Dict[str, Any], config: TripleganaConfig = CONFIG):
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Histograma
    sns.histplot(df['numero'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribución de números (Triple Gana)')

    # Último dígito
    ult = df['ultimo_digito'].value_counts().sort_index()
    sns.barplot(x=list(ult.index), y=list(ult.values), ax=axes[1], palette='rocket')
    axes[1].axhline(y=len(df)/10, color='green', linestyle='--')

    # Media por hora
    hora_means = df.groupby('hora_num')['numero'].mean()
    sns.barplot(x=hora_means.index.astype(str), y=hora_means.values, ax=axes[2], palette='mako')
    axes[2].set_title('Media por hora')

    # Boxplot por hora
    sns.boxplot(x='hora_num', y='numero', data=df, ax=axes[3])
    axes[3].set_title('Distribución por hora')

    # Heatmap correlaciones
    numeric_cols = ['numero', 'hora_num', 'dia_semana', 'mes', 'ultimo_digito', 'suma_digitos']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[4])
    axes[4].set_title('Correlaciones')

    # Top N números
    topn = df['numero'].value_counts().head(config.display_top_n)
    sns.barplot(x=topn.index.astype(str), y=topn.values, ax=axes[5], palette='crest')
    axes[5].set_title(f'Top {config.display_top_n} números')

    plt.tight_layout()
    out_path = Path(config.figures_dir) / 'triplegana_dashboard.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.info("Dashboard guardado en %s", out_path)
    if not config.headless:
        plt.show()
    plt.close(fig)

# -------------------------
# DATASET ML
# -------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def crear_dataset_ml(df: pd.DataFrame, config: TripleganaConfig = CONFIG, test_size: float = 0.2):
    features = list(config.feature_columns)
    X = df[features].fillna(0)
    if config.ml_target == 'ultimo_digito':
        y = df['ultimo_digito'].astype(int)
    elif config.ml_target == 'rango_mil':
        # asegurar que rango_mil exista y sea categórica
        if 'rango_mil' not in df.columns:
            raise ValueError("rango_mil no existe en el DataFrame. Ejecuta preparar_datos primero.")
        y = df['rango_mil'].astype('category').cat.codes
    else:
        y = df['numero'].astype(int)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config.random_state, stratify=stratify)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Dataset ML creado: X_train %s, X_test %s", X_train.shape, X_test.shape)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

# -------------------------
# ORQUESTADOR
# -------------------------
def main():
    logger.info("Iniciando analyzer Triple Gana")
    start = datetime.now(timezone.utc).isoformat()
    df = cargar_datos(CONFIG)
    if df is None or df.empty:
        logger.error("No hay datos para analizar")
        return

    df_prepared = preparar_datos(df, CONFIG)
    if df_prepared is None or df_prepared.empty:
        logger.error("No hay datos válidos después de preparación")
        return

    analysis = analisis_completo(df_prepared, CONFIG)
    visualizar(df_prepared, analysis, CONFIG)

    meta = {
        'timestamp_start': start,
        'timestamp_end': datetime.now(timezone.utc).isoformat(),
        'records': len(df_prepared),
        'config': asdict(CONFIG),
        'summary': {
            'chi2_p': analysis.get('chi2_uniform', {}).get('p'),
            'ultimo_digito_top': analysis.get('ultimo_digito', {}).get('frecuencias')
        }
    }
    _save_json(meta, Path(CONFIG.output_dir) / 'run_metadata_triplegana.json')
    logger.info("Run metadata guardado")

    try:
        crear_dataset_ml(df_prepared, CONFIG)
    except Exception as e:
        logger.warning("No se pudo crear dataset ML: %s", e)

    logger.info("Análisis Triple Gana completado")

if __name__ == "__main__":
    main()
