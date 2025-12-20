"""
analyzer.py - Versión mejorada para Super Gana
Mejoras: CONFIG tipado, logging, modo headless, exportación de resúmenes,
validaciones de supuestos, run metadata, export filas descartadas y dataset ML.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timezone
import logging
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

# Optional imports
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
class AnalyzerConfig:
    db_path: str = 'data/supergana.db'
    table_name: str = 'supergana_resultados'
    output_dir: str = 'data/analysis'
    figures_dir: str = 'data/analysis/figures'
    headless: bool = True                      # Si True no llama plt.show()
    shapiro_max: int = 5000
    chi_bins: int = 20
    rolling_windows: Tuple[int, int] = (7, 30)
    uniformity_alpha: float = 0.05
    random_state: int = 42
    ml_target: str = 'ultimo_digito'           # 'ultimo_digito' | 'rango_mil'
    export_summaries: bool = True
    display_top_n: int = 20
    log_level: str = 'INFO'
    feature_columns: Tuple = ('hora_num', 'dia_semana', 'mes', 'dia_mes',
                              'suma_digitos', 'paridad', 'signo_cod', 'dias_desde_inicio')

CONFIG = AnalyzerConfig()
Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)
Path(CONFIG.figures_dir).mkdir(parents=True, exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=getattr(logging, CONFIG.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(CONFIG.output_dir) / 'analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# UTILIDADES
# -------------------------
def _save_json(obj: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)

def _safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# -------------------------
# CARGA Y VALIDACIÓN
# -------------------------
def cargar_datos(config: AnalyzerConfig = CONFIG) -> Optional[pd.DataFrame]:
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
# PREPARACIÓN Y EXPORTS
# -------------------------
def preparar_datos(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        logger.warning("DataFrame vacío en preparar_datos")
        return None

    df = df.copy()
    # Forzar tipos y detectar filas descartadas
    df['fecha_raw'] = df.get('fecha')
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
    before = len(df)
    # Guardar snapshot de filas con problemas para auditoría
    invalid_mask = df[['fecha', 'numero', 'hora', 'signo']].isna().any(axis=1)
    if invalid_mask.any():
        discarded = df[invalid_mask].copy()
        discarded_path = Path(config.output_dir) / 'filas_descartadas.csv'
        discarded.to_csv(discarded_path, index=False, encoding='utf-8')
        logger.info("Filas descartadas exportadas a %s (count=%d)", discarded_path, len(discarded))
    df = df.dropna(subset=['fecha', 'numero', 'hora', 'signo'])
    dropped = before - len(df)
    if dropped:
        logger.info("Filas descartadas por tipos inválidos: %d", dropped)

    # Hora: parseo robusto
    def parse_hora(h):
        try:
            s = str(h).strip().lower()
            if 'pm' in s:
                hnum = int(''.join(filter(str.isdigit, s)))
                return hnum + 12 if hnum != 12 else 12
            if 'am' in s:
                hnum = int(''.join(filter(str.isdigit, s)))
                return hnum if hnum != 12 else 0
            return int(''.join(filter(str.isdigit, s)))
        except Exception:
            return np.nan

    df['hora_num'] = df['hora'].apply(parse_hora)
    df = df.dropna(subset=['hora_num'])
    df['hora_num'] = df['hora_num'].astype(int)

    # Features básicos
    df['ultimo_digito'] = df['numero'].astype(int) % 10
    df['penultimo_digito'] = (df['numero'].astype(int) // 10) % 10
    df['suma_digitos'] = df['numero'].astype(int).astype(str).apply(lambda s: sum(int(c) for c in s))
    df['paridad'] = df['numero'].astype(int) % 2
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
        bins_fijos = list(range(min_bin, top + 1, 1000))
        labels_fijos = [f'{bins_fijos[i]}-{bins_fijos[i+1]-1}' for i in range(len(bins_fijos)-1)]

        logger.debug(f"bins_fijos: {bins_fijos}")
        logger.debug(f"labels_fijos: {labels_fijos}")

        df['rango_mil'] = pd.cut(df['numero'].astype(int),
                                 bins=bins_fijos,
                                 labels=labels_fijos,
                                 right=False,
                                 include_lowest=True)

        # Opcional: marcar valores fuera de rango (NaN) con etiqueta 'out_of_range'
        df['rango_mil'] = df['rango_mil'].cat.add_categories(['out_of_range']).fillna('out_of_range')
    except Exception as e:
        logger.exception("Error creando rango_mil dinámico: %s", e)
        # Fallback simple: asignar out_of_range a todos si falla
        df['rango_mil'] = 'out_of_range'

    # Temporales
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['dia_mes'] = df['fecha'].dt.day
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dias_desde_inicio'] = (df['fecha'] - df['fecha'].min()).dt.days.astype(int)

    # Guardar dataset preparado
    out_path = Path(config.output_dir) / 'supergana_preparado.csv'
    df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info("Datos preparados guardados en %s", out_path)

    return df

# -------------------------
# TESTS ESTADÍSTICOS Y VALIDACIONES
# -------------------------
def test_normalidad(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    n = len(df)
    sample_size = min(config.shapiro_max, n)
    result = {}
    try:
        sample = df['numero'].sample(sample_size, random_state=config.random_state)
        if n <= config.shapiro_max:
            stat, p = stats.shapiro(sample)
            result['method'] = 'shapiro'
            result['stat'] = float(stat)
            result['p'] = float(p)
            result['normal'] = p > config.uniformity_alpha
        else:
            # Use KS against uniform as fallback for large N (documented)
            numeros_norm = (df['numero'] - df['numero'].min()) / (df['numero'].max() - df['numero'].min())
            stat, p = stats.kstest(numeros_norm, 'uniform')
            result['method'] = 'ks_uniform'
            result['stat'] = float(stat)
            result['p'] = float(p)
            result['normal'] = False
    except Exception as e:
        logger.warning("Error en test_normalidad: %s", e)
        result['error'] = str(e)
    return result

def test_uniformidad_chi2(df: pd.DataFrame, bins: int, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    n = len(df)
    try:
        freq_obs, edges = np.histogram(df['numero'], bins=bins)
        expected = np.array([n / bins] * bins)
        # Warn if expected < 5 in any bin
        if (expected < 5).any():
            logger.warning("Algunas celdas esperadas < 5; chi-square puede no ser válido")
        chi2, p = stats.chisquare(freq_obs, expected)
        return {'chi2': float(chi2), 'p': float(p), 'uniforme': p > config.uniformity_alpha}
    except Exception as e:
        logger.exception("Error en chi2: %s", e)
        return {'error': str(e)}

# -------------------------
# ANÁLISIS Y REPORTES
# -------------------------
def analisis_completo(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    logger.info("Iniciando análisis completo")
    out = {}
    out['n'] = len(df)
    out['range'] = (int(df['numero'].min()), int(df['numero'].max()))
    out['basic_stats'] = {
        'mean': float(df['numero'].mean()),
        'median': float(df['numero'].median()),
        'std': float(df['numero'].std())
    }

    # Normalidad y uniformidad
    out['normality'] = test_normalidad(df, config)
    out['chi2_uniform'] = test_uniformidad_chi2(df, config.chi_bins, config)

    # Último dígito
    ult = df['ultimo_digito'].value_counts().sort_index()
    out['ultimo_digito'] = {
        'frecuencias': ult.to_dict(),
        'porcentajes': (ult / len(df) * 100).round(3).to_dict()
    }
    # Detectar calientes/frios
    esperado = len(df) / 10
    desviacion = ((ult - esperado) / esperado * 100).round(2)
    out['ultimo_digito']['calientes'] = desviacion[desviacion > 10].index.tolist()
    out['ultimo_digito']['frios'] = desviacion[desviacion < -10].index.tolist()

    # Paridad
    par = df['paridad'].value_counts().to_dict()
    out['paridad'] = par

    # Correlaciones
    numeric_cols = ['numero', 'hora_num', 'dia_semana', 'mes', 'ultimo_digito', 'suma_digitos', 'dias_desde_inicio']
    corr = df[numeric_cols].corr()
    out['corr_with_numero'] = corr['numero'].sort_values(ascending=False).to_dict()

    # Temporal: Kruskal por hora (no paramétrico)
    try:
        groups = [g['numero'].values for _, g in df.groupby('hora_num')]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            out['kruskal_hora'] = {'h': float(h), 'p': float(p), 'significativo': p < config.uniformity_alpha}
    except Exception as e:
        logger.warning("Kruskal por hora falló: %s", e)

    # Autocorrelación (si disponible)
    if sm_acf is not None:
        try:
            acf_vals = sm_acf(df['numero'].values, nlags=20, fft=False)
            out['acf'] = [float(x) for x in acf_vals]
        except Exception as e:
            logger.debug("ACF falló: %s", e)
            out['acf'] = None
    else:
        out['acf'] = None

    # Runs test (si disponible)
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
        _safe_mkdir(config.output_dir)
        pd.Series(out['ultimo_digito']['frecuencias']).to_csv(Path(config.output_dir) / 'supergana_ultimo_digito.csv')
        pd.Series(out['paridad']).to_csv(Path(config.output_dir) / 'supergana_paridad.csv')
        corr.to_csv(Path(config.output_dir) / 'supergana_corr.csv')
        _save_json(out, Path(config.output_dir) / 'supergana_tests.json')
        logger.info("Resúmenes exportados a %s", config.output_dir)

    return out

# -------------------------
# VISUALIZACIONES (HEADLESS SAFE)
# -------------------------
def visualizar(df: pd.DataFrame, analysis: Dict[str, Any], config: AnalyzerConfig = CONFIG):
    sns.set_style('darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Histograma
    sns.histplot(df['numero'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribución de números')

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
    out_path = Path(config.figures_dir) / 'supergana_dashboard.png'
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

def crear_dataset_ml(df: pd.DataFrame, config: AnalyzerConfig = CONFIG, test_size: float = 0.2):
    features = list(config.feature_columns)
    X = df[features].fillna(0)
    if config.ml_target == 'ultimo_digito':
        y = df['ultimo_digito'].astype(int)
    elif config.ml_target == 'rango_mil':
        y = df['rango_mil'].astype('category').cat.codes
    else:
        y = df['numero'].astype(int)

    stratify = y if len(y.unique()) > 1 else None
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
    logger.info("Iniciando analyzer mejorado")
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

    # Guardar metadata del run
    meta = {
        'timestamp': start,
        'finished': datetime.now(timezone.utc).isoformat(),
        'records': len(df_prepared),
        'config': asdict(CONFIG),
        'analysis_summary': {
            'chi2_p': analysis.get('chi2_uniform', {}).get('p'),
            'normality_p': analysis.get('normality', {}).get('p')
        }
    }
    _save_json(meta, Path(CONFIG.output_dir) / 'run_metadata.json')
    logger.info("Run metadata guardado")

    # Crear dataset ML de ejemplo
    try:
        crear_dataset_ml(df_prepared, CONFIG)
    except Exception as e:
        logger.warning("No se pudo crear dataset ML: %s", e)

    logger.info("Análisis completado")

if __name__ == "__main__":
    main()
