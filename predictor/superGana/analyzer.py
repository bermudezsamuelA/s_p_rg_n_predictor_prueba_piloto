# analyzer_supergana.py — Analyzer mejorado para Super Gana con ML
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime, timezone
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy, binomtest

# Optional statsmodels
try:
    from statsmodels.tsa.stattools import acf as sm_acf
    from statsmodels.sandbox.stats.runs import runstest_1samp
    from statsmodels.stats.diagnostic import acorr_ljungbox
    sm_modules_available = True
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Statsmodels no disponible completamente: {e}")
    sm_acf = None
    runstest_1samp = None
    acorr_ljungbox = None
    sm_modules_available = False

# -------------------------
# CONFIG
# -------------------------
@dataclass
class AnalyzerConfig:
    db_path: str = 'data/supergana.db'
    table_name: str = 'supergana_resultados'
    output_dir: str = 'data/analysis'
    figures_dir: str = 'data/analysis/figures'
    headless: bool = True
    shapiro_max: int = 5000
    chi_bins: int = 20
    rolling_windows: Tuple[int, int] = (7, 30)
    uniformity_alpha: float = 0.05
    random_state: int = 42
    ml_target: str = 'ultimo_digito'  # 'ultimo_digito' | 'rango_mil' | 'numero_int'
    export_summaries: bool = True
    display_top_n: int = 20
    log_level: str = 'INFO'
    feature_columns: Tuple = ('hora_num', 'dia_semana', 'mes', 'dia_mes',
                              'suma_digitos', 'paridad', 'signo_cod', 'dias_desde_inicio')
    # Nuevos parámetros
    hot_cold_threshold: float = 10.0  # Umbral para dígitos calientes/fríos (%)
    n_simulations: int = 1000  # Para tests Monte Carlo
    entropy_threshold: float = 0.95  # Umbral para alta entropía
    ljung_box_lags: int = 10  # Lags para test Ljung-Box
    # Parámetros ML
    crear_ml_dataset: bool = True  # Crear dataset para machine learning
    test_size: float = 0.2  # Proporción para test split
    ml_export_format: str = 'csv'  # 'csv', 'npz', o 'both'

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
        logging.FileHandler(Path(CONFIG.output_dir) / 'analyzer_supergana.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# UTILIDADES
# -------------------------
def _save_json(obj: Any, path: Path):
    """Guarda objeto como JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)

def _safe_mkdir(path: Path):
    """Crea directorio si no existe"""
    path.mkdir(parents=True, exist_ok=True)

# -------------------------
# TESTS AVANZADOS (optimizados)
# -------------------------
def test_entropia_secuencia(secuencia: np.ndarray, num_categories: int = 10000) -> Dict[str, Any]:
    """Calcula la entropía de Shannon de una secuencia"""
    try:
        valores, counts = np.unique(secuencia, return_counts=True)
        frecuencias = counts / counts.sum()
        ent = entropy(frecuencias, base=2)
        ent_max = np.log2(num_categories)
        ratio = ent / ent_max
        
        return {
            'entropia': float(ent),
            'entropia_maxima': float(ent_max),
            'ratio': float(ratio),
            'alta_entropia': bool(ratio > CONFIG.entropy_threshold),
            'interpretacion': f"{ratio:.1%} de entropía máxima"
        }
    except Exception as e:
        logger.error(f"Error en test de entropía: {e}")
        return {'error': str(e)}

def test_ljung_box(secuencia: np.ndarray, lags: int = None) -> Optional[Dict[str, Any]]:
    """Test de Ljung-Box para independencia serial"""
    if acorr_ljungbox is None:
        logger.warning("acorr_ljungbox no disponible")
        return None
    
    try:
        lags = lags or CONFIG.ljung_box_lags
        result = acorr_ljungbox(secuencia, lags=lags, return_df=True)
        p_values = result['lb_pvalue'].values
        estadisticos = result['lb_stat'].values
        hay_autocorrelacion = any(p < CONFIG.uniformity_alpha for p in p_values)
        
        return {
            'lags': list(range(1, lags + 1)),
            'estadisticos': [float(x) for x in estadisticos],
            'p_values': [float(p) for p in p_values],
            'independiente': bool(not hay_autocorrelacion),
            'primer_lag_significativo': next((i+1 for i, p in enumerate(p_values) if p < CONFIG.uniformity_alpha), None)
        }
    except Exception as e:
        logger.error(f"Error en test Ljung-Box: {e}")
        return None

def test_uniformidad_digitos(secuencia: np.ndarray, n_categorias: int = 10) -> Dict[str, Any]:
    """Test de uniformidad para dígitos usando chi-cuadrado"""
    try:
        valores, counts = np.unique(secuencia, return_counts=True)
        freq_obs = np.zeros(n_categorias)
        
        for val, count in zip(valores, counts):
            if 0 <= val < n_categorias:
                freq_obs[val] = count
        
        n_total = counts.sum()
        freq_esp = np.array([n_total / n_categorias] * n_categorias)
        chi2, p = stats.chisquare(freq_obs, freq_esp)
        
        # Calcular desviaciones
        desviaciones = {}
        for i in range(n_categorias):
            obs = freq_obs[i]
            esp = freq_esp[i]
            if esp > 0:
                desviacion_pct = (obs - esp) / esp * 100
                desviaciones[str(i)] = {
                    'observado': float(obs),
                    'esperado': float(esp),
                    'desviacion_pct': float(desviacion_pct),
                    'es_caliente': bool(desviacion_pct > CONFIG.hot_cold_threshold),
                    'es_frio': bool(desviacion_pct < -CONFIG.hot_cold_threshold)
                }
        
        digitos_calientes = [str(i) for i, d in desviaciones.items() if d['es_caliente']]
        digitos_frios = [str(i) for i, d in desviaciones.items() if d['es_frio']]
        
        return {
            'chi2': float(chi2),
            'p_value': float(p),
            'uniforme': bool(p > CONFIG.uniformity_alpha),
            'desviaciones': desviaciones,
            'digitos_calientes': digitos_calientes,
            'digitos_frios': digitos_frios,
            'max_desviacion_pct': float(max(abs(d['desviacion_pct']) for d in desviaciones.values()))
        }
    except Exception as e:
        logger.error(f"Error en test uniformidad dígitos: {e}")
        return {'error': str(e)}

# -------------------------
# CARGA Y VALIDACIÓN
# -------------------------
def cargar_datos(config: AnalyzerConfig = CONFIG) -> Optional[pd.DataFrame]:
    """Carga datos desde la base de datos SQLite"""
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
# PREPARACIÓN (VECTORIZADA)
# -------------------------
def preparar_datos(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Optional[pd.DataFrame]:
    """Prepara y limpia los datos para análisis"""
    if df is None or df.empty:
        logger.warning("DataFrame vacío en preparar_datos")
        return None

    df = df.copy()
    df['fecha_raw'] = df.get('fecha')
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')

    # Filtrar filas inválidas
    before = len(df)
    invalid_mask = df[['fecha', 'numero', 'hora', 'signo']].isna().any(axis=1)
    if invalid_mask.any():
        discarded = df[invalid_mask].copy()
        discarded_path = Path(config.output_dir) / 'filas_descartadas_supergana.csv'
        discarded.to_csv(discarded_path, index=False, encoding='utf-8')
        logger.info("Filas descartadas exportadas a %s (count=%d)", discarded_path, len(discarded))

    df = df.dropna(subset=['fecha', 'numero', 'hora', 'signo'])
    dropped = before - len(df)
    if dropped:
        logger.info("Filas descartadas por tipos inválidos: %d", dropped)

    # Hora vectorizada (mejorada)
    def parse_hora_vectorizado(s):
        s = s.astype(str).str.lower().str.strip()
        
        # Extraer números
        horas = s.str.extract(r'(\d+)', expand=False)
        horas = pd.to_numeric(horas, errors='coerce')
        
        # Ajustar para AM/PM
        is_pm = s.str.contains('pm')
        is_am = s.str.contains('am')
        
        # Casos especiales
        horas = horas.where(~(is_am & (horas == 12)), 0)
        horas = horas.where(~(is_pm & (horas != 12)), horas + 12)
        horas = horas.where(~(is_pm & (horas == 12)), 12)
        
        return horas.fillna(-1).astype(int)
    
    df['hora_num'] = parse_hora_vectorizado(df['hora'])
    
    # Eliminar filas con hora inválida
    df = df[df['hora_num'] >= 0]

    # Unificar columna numérica
    df['numero_int'] = df['numero'].astype(int)

    # Vectorizar extracción de 4 dígitos (optimizada)
    def extraer_digitos_vectorizado(serie: pd.Series) -> pd.DataFrame:
        """Extrae dígitos de manera vectorizada"""
        str_nums = serie.apply(lambda x: f"{int(x):04d}")
        digits = str_nums.str.extract(r'(\d)(\d)(\d)(\d)').astype(int)
        digits.columns = ['primer_digito', 'segundo_digito', 'tercer_digito', 'cuarto_digito']
        return digits
    
    digits = extraer_digitos_vectorizado(df['numero_int'])
    df = pd.concat([df.reset_index(drop=True), digits.reset_index(drop=True)], axis=1)

    # Último dígito
    df['ultimo_digito'] = df['cuarto_digito']

    # Otras features (vectorizadas)
    df['suma_digitos'] = df['numero_int'].astype(str).str.extractall(r'(\d)').groupby(level=0)[0].astype(int).sum()
    df['paridad'] = df['numero_int'] % 2
    df['signo'] = df['signo'].astype(str)
    df['signo_cod'] = pd.factorize(df['signo'])[0].astype('int8')

    # Rangos por mil (dinámico, defensivo)
    try:
        min_num = int(df['numero_int'].min())
        max_num = int(df['numero_int'].max())
        min_bin = (min_num // 1000) * 1000
        top = ((max_num // 1000) + 1) * 1000
        if top <= min_bin:
            top = min_bin + 1000
        bins_fijos = list(range(min_bin, top + 1, 1000))
        labels_fijos = [f'{bins_fijos[i]:04d}-{bins_fijos[i+1]-1:04d}' 
                       for i in range(len(bins_fijos)-1)]
        df['rango_mil'] = pd.cut(df['numero_int'], bins=bins_fijos, 
                                labels=labels_fijos, right=False, include_lowest=True)
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

    # Guardar dataset preparado
    out_path = Path(config.output_dir) / 'supergana_preparado.csv'
    df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info("Datos preparados guardados en %s (registros: %d)", out_path, len(df))
    
    return df

# -------------------------
# CHI2 MEJORADO Y ANÁLISIS
# -------------------------
def test_uniformidad_chi2(df: pd.DataFrame, bins: int, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    """Test chi-cuadrado para uniformidad (mejorado)"""
    n = len(df)
    try:
        nums = df['numero_int'].dropna().astype(int).values
        freq_obs, edges = np.histogram(nums, bins=bins)
        expected = np.array([len(nums) / bins] * bins)

        # Validaciones
        expected_issue = (expected < 5).any()
        observed_issue = (freq_obs < 5).any()
        if expected_issue or observed_issue:
            logger.warning("Chi2: algunas celdas esperadas u observadas < 5; chi-square puede no ser válido")

        # Calcular chi2
        chi2, p = stats.chisquare(freq_obs, expected)
        
        return {
            'chi2': float(chi2),
            'p': float(p),
            'uniforme': bool(p > config.uniformity_alpha),
            'grados_libertad': bins - 1,
            'frecuencias_observadas': freq_obs.tolist(),
            'frecuencias_esperadas': expected.tolist(),
            'edges': edges.tolist(),
            'n_celdas_problematicas': int((freq_obs < 5).sum()),
            'advertencia_celdas': expected_issue or observed_issue
        }
    except Exception as e:
        logger.exception("Error en chi2: %s", e)
        return {'error': str(e)}

def test_normalidad(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    """Test de normalidad (Shapiro-Wilk o KS)"""
    n = len(df)
    result = {}
    try:
        sample_size = min(config.shapiro_max, n)
        sample = df['numero_int'].sample(sample_size, random_state=config.random_state)
        
        if n <= config.shapiro_max:
            stat, p = stats.shapiro(sample)
            result['method'] = 'shapiro'
        else:
            # Normalizar para test KS
            numeros_norm = (df['numero_int'] - df['numero_int'].min()) / (df['numero_int'].max() - df['numero_int'].min())
            stat, p = stats.kstest(numeros_norm, 'uniform')
            result['method'] = 'ks_uniform'
        
        result['stat'] = float(stat)
        result['p'] = float(p)
        result['normal'] = bool(p > config.uniformity_alpha)
        
    except Exception as e:
        logger.warning("Error en test_normalidad: %s", e)
        result['error'] = str(e)
    
    return result

# -------------------------
# DATASET PARA MACHINE LEARNING
# -------------------------
def crear_dataset_ml(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Optional[Dict[str, Any]]:
    """
    Crea dataset para machine learning a partir de datos preparados.
    
    Returns:
        Dict con datasets y metadatos, o None si hay error
    """
    try:
        # Importar aquí para no forzar dependencia si no se usa ML
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import pickle
        
        logger.info("Creando dataset para Machine Learning...")
        
        # Seleccionar features
        features = list(config.feature_columns)
        features_disponibles = [f for f in features if f in df.columns]
        
        if len(features_disponibles) == 0:
            logger.warning("No hay features disponibles para ML")
            return None
        
        X = df[features_disponibles].fillna(0)
        
        # Seleccionar target según configuración
        if config.ml_target == 'ultimo_digito':
            y = df['ultimo_digito'].astype(int)
            problem_type = 'classification'
            n_classes = 10
            logger.info(f"Target: último dígito (0-9), {n_classes} clases")
            
        elif config.ml_target == 'rango_mil':
            # Convertir categorías a códigos numéricos
            le = LabelEncoder()
            y = le.fit_transform(df['rango_mil'].astype(str))
            problem_type = 'classification'
            n_classes = len(le.classes_)
            logger.info(f"Target: rango por mil, {n_classes} clases")
            
        elif config.ml_target == 'numero_int':
            y = df['numero_int'].astype(int)
            problem_type = 'regression'
            n_classes = None
            logger.info(f"Target: número entero completo (regresión)")
            
        else:
            logger.error(f"Target ML desconocido: {config.ml_target}")
            return None
        
        # Verificar si hay suficientes muestras
        if len(df) < 100:
            logger.warning(f"Dataset pequeño para ML: {len(df)} muestras")
        
        # Split estratificado para clasificación, random para regresión
        if problem_type == 'classification' and n_classes > 1:
            stratify = y
        else:
            stratify = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.test_size, 
            random_state=config.random_state,
            stratify=stratify
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Metadatos del dataset
        metadata = {
            'n_samples': len(df),
            'n_features': len(features_disponibles),
            'features': features_disponibles,
            'target': config.ml_target,
            'problem_type': problem_type,
            'n_classes': n_classes,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_ratio': config.test_size,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Exportar según formato configurado
        if config.ml_export_format in ['csv', 'both']:
            # Guardar datasets en CSV
            pd.DataFrame(X_train_scaled, columns=features_disponibles).to_csv(
                Path(config.output_dir) / 'ml_X_train.csv', index=False
            )
            pd.DataFrame(X_test_scaled, columns=features_disponibles).to_csv(
                Path(config.output_dir) / 'ml_X_test.csv', index=False
            )
            pd.DataFrame(y_train, columns=['target']).to_csv(
                Path(config.output_dir) / 'ml_y_train.csv', index=False
            )
            pd.DataFrame(y_test, columns=['target']).to_csv(
                Path(config.output_dir) / 'ml_y_test.csv', index=False
            )
            logger.info("Dataset ML exportado como CSV")
        
        if config.ml_export_format in ['npz', 'both']:
            # Guardar en formato NPZ (más eficiente)
            np.savez_compressed(
                Path(config.output_dir) / 'ml_dataset.npz',
                X_train=X_train_scaled,
                X_test=X_test_scaled,
                y_train=y_train,
                y_test=y_test,
                features=features_disponibles,
                target_name=config.ml_target
            )
            logger.info("Dataset ML exportado como NPZ")
        
        # Guardar scaler
        with open(Path(config.output_dir) / 'ml_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Guardar metadata
        _save_json(metadata, Path(config.output_dir) / 'ml_metadata.json')
        
        logger.info(f"Dataset ML creado exitosamente: "
                   f"X_train {X_train_scaled.shape}, X_test {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'features': features_disponibles,
            'metadata': metadata
        }
        
    except ImportError as e:
        logger.error(f"Faltan dependencias para ML: {e}")
        logger.info("Instala scikit-learn: pip install scikit-learn")
        return None
    except Exception as e:
        logger.exception(f"Error creando dataset ML: {e}")
        return None

# -------------------------
# ANÁLISIS COMPLETO
# -------------------------
def analisis_completo(df: pd.DataFrame, config: AnalyzerConfig = CONFIG) -> Dict[str, Any]:
    """Análisis completo con todos los tests"""
    logger.info("Iniciando análisis completo Super Gana")
    out: Dict[str, Any] = {}
    
    # Información básica
    out['n'] = len(df)
    out['range'] = (int(df['numero_int'].min()), int(df['numero_int'].max()))
    out['basic_stats'] = {
        'mean': float(df['numero_int'].mean()),
        'median': float(df['numero_int'].median()),
        'std': float(df['numero_int'].std()),
        'skew': float(df['numero_int'].skew()),
        'kurtosis': float(df['numero_int'].kurtosis()),
        'q1': float(df['numero_int'].quantile(0.25)),
        'q3': float(df['numero_int'].quantile(0.75)),
        'iqr': float(df['numero_int'].quantile(0.75) - df['numero_int'].quantile(0.25))
    }

    # 1. Tests de distribución básicos
    out['normality'] = test_normalidad(df, config)
    out['chi2_uniform'] = test_uniformidad_chi2(df, config.chi_bins, config)
    
    # 2. Tests de aleatoriedad avanzados
    out['entropia'] = test_entropia_secuencia(df['numero_int'].values, num_categories=10000)
    
    # 3. Tests de independencia temporal
    if sm_modules_available:
        out['ljung_box'] = test_ljung_box(df['numero_int'].values)
    
    # 4. Análisis por posición de dígito (CRÍTICO)
    out['analisis_digitos'] = {}
    posiciones = ['primer_digito', 'segundo_digito', 'tercer_digito', 'cuarto_digito']
    
    for pos in posiciones:
        if pos in df.columns:
            secuencia = df[pos].values
            test_result = test_uniformidad_digitos(secuencia)
            
            # Información adicional
            frecuencias = df[pos].value_counts().sort_index()
            porcentajes = (frecuencias / len(df) * 100).round(3)
            
            out['analisis_digitos'][pos] = {
                'frecuencias': frecuencias.to_dict(),
                'porcentajes': porcentajes.to_dict(),
                'test_uniformidad': test_result,
                'esperado': 10.0,  # 10% para cada dígito 0-9
                'max_desviacion': float(max(abs(porcentajes.get(d, 0) - 10.0) for d in range(10)) if not porcentajes.empty else 0)
            }
    
    # 5. Último dígito (información adicional)
    ult = df['ultimo_digito'].value_counts().sort_index()
    out['ultimo_digito'] = {
        'frecuencias': ult.to_dict(),
        'porcentajes': (ult / len(df) * 100).round(3).to_dict()
    }
    
    esperado = len(df) / 10
    desviacion = ((ult - esperado) / esperado * 100).round(2)
    out['ultimo_digito']['calientes'] = desviacion[desviacion > config.hot_cold_threshold].index.tolist()
    out['ultimo_digito']['frios'] = desviacion[desviacion < -config.hot_cold_threshold].index.tolist()
    out['ultimo_digito']['desviaciones'] = desviacion.to_dict()
    
    # 6. Paridad y signos
    out['paridad'] = df['paridad'].value_counts().to_dict()
    out['signos'] = df['signo'].value_counts().to_dict()
    
    # Test binomial para paridad
    n_par = out['paridad'].get(0, 0)
    n_total = len(df)
    binom_result = binomtest(n_par, n_total, 0.5)
    out['test_binomial_paridad'] = {
        'p': float(binom_result.pvalue),
        'balanceado': bool(binom_result.pvalue > config.uniformity_alpha),
        'proporcion_pares': float(n_par / n_total)
    }
    
    # 7. Correlaciones
    numeric_cols = ['numero_int', 'hora_num', 'dia_semana', 'mes', 
                    'ultimo_digito', 'suma_digitos', 'dias_desde_inicio']
    disponibles = [col for col in numeric_cols if col in df.columns]
    
    if len(disponibles) > 1:
        corr = df[disponibles].corr()
        out['corr_with_numero'] = corr['numero_int'].sort_values(ascending=False).to_dict()
        out['correlation_matrix'] = corr.to_dict()
    
    # 8. Tests temporales
    try:
        groups = [g['numero_int'].values for _, g in df.groupby('hora_num')]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            out['kruskal_hora'] = {
                'h': float(h), 
                'p': float(p), 
                'significativo': bool(p < config.uniformity_alpha)
            }
    except Exception as e:
        logger.warning("Kruskal por hora falló: %s", e)
    
    # 9. Autocorrelación
    if sm_acf is not None:
        try:
            acf_vals = sm_acf(df['numero_int'].values, nlags=20, fft=False)
            out['acf'] = [float(x) for x in acf_vals]
        except Exception as e:
            logger.debug("ACF falló: %s", e)
            out['acf'] = None
    
    # 10. Runs test
    if runstest_1samp is not None:
        try:
            stat, p = runstest_1samp(df['numero_int'])
            out['runs_test'] = {
                'stat': float(stat), 
                'p': float(p), 
                'aleatorio': bool(p > config.uniformity_alpha)
            }
        except Exception as e:
            logger.debug("runstest_1samp falló: %s", e)
            out['runs_test'] = None
    
    # 11. Exportar resúmenes
    if config.export_summaries:
        _safe_mkdir(Path(config.output_dir))
        
        # Exportar tests como CSV
        tests_data = []
        tests_data.append(('chi2_uniform', out.get('chi2_uniform', {}).get('p'), 
                          out.get('chi2_uniform', {}).get('uniforme', False)))
        tests_data.append(('shapiro', out.get('normality', {}).get('p'), 
                          out.get('normality', {}).get('normal', False)))
        tests_data.append(('entropia_ratio', out.get('entropia', {}).get('ratio'), 
                          out.get('entropia', {}).get('alta_entropia', False)))
        tests_data.append(('binomial_paridad', out.get('test_binomial_paridad', {}).get('p'), 
                          out.get('test_binomial_paridad', {}).get('balanceado', False)))
        
        tests_df = pd.DataFrame(tests_data, columns=['test', 'p_value', 'resultado'])
        tests_df.to_csv(Path(config.output_dir) / 'supergana_tests_summary.csv', index=False)
        
        # Exportar análisis de dígitos por posición
        for pos in posiciones:
            if pos in out.get('analisis_digitos', {}):
                dig_df = pd.DataFrame.from_dict(
                    out['analisis_digitos'][pos]['frecuencias'], 
                    orient='index', 
                    columns=['frecuencia']
                )
                dig_df['porcentaje'] = (dig_df['frecuencia'] / dig_df['frecuencia'].sum() * 100).round(2)
                dig_df.to_csv(Path(config.output_dir) / f'{pos}_analysis.csv')
        
        # Exportar JSON completo
        _save_json(out, Path(config.output_dir) / 'supergana_tests_completos.json')
        logger.info("Resúmenes exportados a %s", config.output_dir)
    
    # 12. Generar reporte ejecutivo
    out['reporte_ejecutivo'] = generar_reporte_ejecutivo(out, config)
    
    return out

def generar_reporte_ejecutivo(analisis: Dict, config: AnalyzerConfig) -> Dict:
    """Genera reporte ejecutivo con conclusiones"""
    reporte = {
        'fecha_analisis': datetime.now(timezone.utc).isoformat(),
        'muestras': analisis.get('n', 0),
        'conclusiones_principales': [],
        'alertas': [],
        'recomendaciones': []
    }
    
    # Evaluar uniformidad
    chi2_result = analisis.get('chi2_uniform', {})
    if chi2_result and 'uniforme' in chi2_result:
        if chi2_result['uniforme']:
            reporte['conclusiones_principales'].append("✅ Distribución uniforme (aleatoria)")
        else:
            reporte['conclusiones_principales'].append("⚠️ Distribución NO uniforme (posible sesgo)")
            reporte['alertas'].append(f"Chi² p-value: {chi2_result.get('p', 'N/A')}")
    
    # Evaluar entropía
    entropia_result = analisis.get('entropia', {})
    if entropia_result and 'alta_entropia' in entropia_result:
        if entropia_result['alta_entropia']:
            reporte['conclusiones_principales'].append("✅ Alta entropía (alta aleatoriedad)")
        else:
            reporte['conclusiones_principales'].append("⚠️ Baja entropía (posible patrón)")
            reporte['alertas'].append(f"Ratio entropía: {entropia_result.get('ratio', 'N/A'):.3f}")
    
    # Evaluar dígitos por posición
    analisis_digitos = analisis.get('analisis_digitos', {})
    for pos, datos in analisis_digitos.items():
        test_result = datos.get('test_uniformidad', {})
        if test_result and 'uniforme' in test_result:
            if not test_result['uniforme']:
                calientes = test_result.get('digitos_calientes', [])
                frios = test_result.get('digitos_frios', [])
                if calientes:
                    reporte['alertas'].append(f"{pos.replace('_', ' ').title()} calientes: {calientes}")
                if frios:
                    reporte['alertas'].append(f"{pos.replace('_', ' ').title()} fríos: {frios}")
    
    # Recomendaciones
    if not chi2_result.get('uniforme', True):
        reporte['recomendaciones'].append(
            "Considerar estrategias basadas en dígitos sub/sobrerrepresentados"
        )
    
    if analisis.get('test_binomial_paridad', {}).get('balanceado', True):
        reporte['recomendaciones'].append("Paridad balanceada - no hay ventaja en pares/impares")
    
    return reporte

# -------------------------
# VISUALIZACIONES (12 gráficas)
# -------------------------
def visualizar(df: pd.DataFrame, analysis: Dict[str, Any], config: AnalyzerConfig = CONFIG):
    """Visualizaciones completas con 12 gráficas"""
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = [20, 16]
    plt.rcParams['figure.autolayout'] = True
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Distribución de números
    ax1 = plt.subplot(3, 4, 1)
    sns.histplot(df['numero_int'], bins=50, kde=True, ax=ax1, color='skyblue')
    ax1.axvline(df['numero_int'].mean(), color='red', linestyle='--', 
                label=f'Media: {df["numero_int"].mean():.0f}')
    ax1.axvline(df['numero_int'].median(), color='green', linestyle='--', 
                label=f'Mediana: {df["numero_int"].median():.0f}')
    ax1.set_title('Distribución de números (Super Gana)', fontweight='bold')
    ax1.legend()
    
    # 2. Boxplot por hora
    ax2 = plt.subplot(3, 4, 2)
    sns.boxplot(x='hora_num', y='numero_int', data=df, ax=ax2)
    ax2.set_title('Distribución por hora', fontweight='bold')
    ax2.set_xlabel('Hora')
    ax2.set_ylabel('Número')
    
    # 3-6. Distribución por posición de dígito
    posiciones = ['primer_digito', 'segundo_digito', 'tercer_digito', 'cuarto_digito']
    for i, pos in enumerate(posiciones, 3):
        ax = plt.subplot(3, 4, i)
        if pos in df.columns:
            frec = df[pos].value_counts().sort_index()
            colors = ['red' if d in analysis.get('analisis_digitos', {}).get(pos, {}).get('test_uniformidad', {}).get('digitos_calientes', []) 
                     else 'blue' if d in analysis.get('analisis_digitos', {}).get(pos, {}).get('test_uniformidad', {}).get('digitos_frios', []) 
                     else 'gray' for d in frec.index]
            
            bars = ax.bar(frec.index.astype(str), frec.values, color=colors, edgecolor='black')
            ax.axhline(y=len(df)/10, color='green', linestyle='--', label='Esperado (10%)')
            ax.set_title(f'{pos.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Dígito')
            ax.set_ylabel('Frecuencia')
            ax.legend()
    
    # 7. Paridad
    ax7 = plt.subplot(3, 4, 7)
    paridad_counts = df['paridad'].value_counts()
    labels = ['Pares' if x == 0 else 'Impares' for x in paridad_counts.index]
    colors = ['lightblue', 'lightcoral']
    wedges, texts, autotexts = ax7.pie(paridad_counts.values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax7.set_title('Distribución Par/Impar', fontweight='bold')
    
    # 8. Entropía
    ax8 = plt.subplot(3, 4, 8)
    if 'entropia' in analysis:
        ent_data = analysis['entropia']
        categories = ['Entropía Observada', 'Entropía Máxima']
        values = [ent_data.get('entropia', 0), ent_data.get('entropia_maxima', 0)]
        colors = ['lightblue' if ent_data.get('alta_entropia', False) else 'lightcoral', 'gray']
        
        bars = ax8.bar(categories, values, color=colors, edgecolor='black')
        ax8.set_title('Análisis de Entropía', fontweight='bold')
        ax8.set_ylabel('Bits')
        ax8.set_ylim(0, ent_data.get('entropia_maxima', 1) * 1.2)
        ax8.text(0.5, max(values) * 1.05, f'Ratio: {ent_data.get("ratio", 0):.3f}', 
                ha='center', fontweight='bold')
    
    # 9. Correlaciones
    ax9 = plt.subplot(3, 4, 9)
    numeric_cols = ['numero_int', 'hora_num', 'dia_semana', 'mes', 'ultimo_digito', 'suma_digitos']
    disponibles = [col for col in numeric_cols if col in df.columns]
    
    if len(disponibles) > 1:
        corr = df[disponibles].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax9, 
                   cbar_kws={'shrink': 0.8})
        ax9.set_title('Matriz de Correlaciones', fontweight='bold')
    
    # 10. Top números
    ax10 = plt.subplot(3, 4, 10)
    topn = df['numero_int'].value_counts().head(config.display_top_n)
    sns.barplot(x=topn.index.astype(str), y=topn.values, ax=ax10, palette='crest')
    ax10.set_title(f'Top {config.display_top_n} números', fontweight='bold')
    ax10.set_xlabel('Número')
    ax10.set_ylabel('Frecuencia')
    ax10.tick_params(axis='x', rotation=45)
    
    # 11. Evolución temporal
    ax11 = plt.subplot(3, 4, 11)
    if 'fecha' in df.columns and len(df) > 30:
        df_sorted = df.sort_values('fecha').reset_index(drop=True)
        df_sorted['media_movil_30'] = df_sorted['numero_int'].rolling(30).mean()
        ax11.plot(df_sorted.index, df_sorted['media_movil_30'], 'r-', linewidth=2)
        ax11.set_title('Media Móvil 30 días', fontweight='bold')
        ax11.set_xlabel('Índice')
        ax11.set_ylabel('Número (media móvil)')
    
    # 12. Resumen estadístico
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    stats_text = f"""
    Estadísticas Clave:
    -------------------
    Muestras: {len(df):,}
    Rango: {df['numero_int'].min()} - {df['numero_int'].max()}
    Media: {df['numero_int'].mean():.0f}
    Mediana: {df['numero_int'].median():.0f}
    Std: {df['numero_int'].std():.0f}
    
    Tests de Aleatoriedad:
    ---------------------
    Chi² Uniformidad: {'✓' if analysis.get('chi2_uniform', {}).get('uniforme', False) else '✗'}
    Entropía: {analysis.get('entropia', {}).get('ratio', 0):.3f}
    Runs Test: {'✓' if analysis.get('runs_test', {}).get('aleatorio', False) else '✗'}
    """
    
    ax12.text(0.1, 0.95, stats_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ANÁLISIS COMPLETO SUPER GANA - DASHBOARD AVANZADO', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Guardar figura
    out_path = Path(config.figures_dir) / 'supergana_dashboard_avanzado.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.info("Dashboard avanzado guardado en %s", out_path)
    
    if not config.headless:
        plt.show()
    plt.close(fig)
    
    # Crear gráfico adicional para dígitos por posición
    visualizar_digitos_detalle(df, analysis, config)

def visualizar_digitos_detalle(df: pd.DataFrame, analysis: Dict[str, Any], config: AnalyzerConfig):
    """Visualización especializada para análisis de dígitos por posición"""
    posiciones = ['primer_digito', 'segundo_digito', 'tercer_digito', 'cuarto_digito']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, pos in enumerate(posiciones):
        if i >= len(axes):
            break
            
        if pos in df.columns:
            ax = axes[i]
            frec = df[pos].value_counts().sort_index()
            porcentajes = (frec / len(df) * 100).round(2)
            
            # Colores según desviación
            colors = []
            for dig in frec.index:
                pct = porcentajes[dig]
                if pct > 10 + config.hot_cold_threshold:
                    colors.append('red')  # Muy caliente
                elif pct > 10:
                    colors.append('orange')  # Caliente
                elif pct < 10 - config.hot_cold_threshold:
                    colors.append('blue')  # Muy frío
                elif pct < 10:
                    colors.append('lightblue')  # Frío
                else:
                    colors.append('gray')  # Normal
            
            bars = ax.bar(frec.index.astype(str), frec.values, color=colors, edgecolor='black')
            ax.axhline(y=len(df)/10, color='green', linestyle='--', linewidth=2, label='Esperado')
            
            # Añadir porcentajes
            for bar, pct in zip(bars, porcentajes.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title(f'{pos.replace("_", " ").title()}\nDistribución por Dígito', 
                        fontweight='bold')
            ax.set_xlabel('Dígito')
            ax.set_ylabel('Frecuencia')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('ANÁLISIS DETALLADO DE DÍGITOS POR POSICIÓN', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = Path(config.figures_dir) / 'digitos_por_posicion_detalle.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.info("Gráfico dígitos por posición guardado en %s", out_path)
    
    if not config.headless:
        plt.show()
    plt.close(fig)

# -------------------------
# ORQUESTADOR
# -------------------------
def main():
    """Función principal"""
    logger.info("=" * 60)
    logger.info("INICIANDO ANALYZER SUPER GANA CON ML")
    logger.info("=" * 60)
    
    start = datetime.now(timezone.utc).isoformat()
    
    # Cargar datos
    df = cargar_datos(CONFIG)
    if df is None or df.empty:
        logger.error("No hay datos para analizar")
        return
    
    # Preparar datos
    df_prepared = preparar_datos(df, CONFIG)
    if df_prepared is None or df_prepared.empty:
        logger.error("No hay datos válidos después de preparación")
        return
    
    logger.info(f"Datos preparados: {len(df_prepared)} registros válidos")
    
    # Análisis completo
    analysis = analisis_completo(df_prepared, CONFIG)
    
    # Visualizaciones
    visualizar(df_prepared, analysis, CONFIG)
    
    # Crear dataset ML si está configurado
    ml_result = None
    if CONFIG.crear_ml_dataset:
        ml_result = crear_dataset_ml(df_prepared, CONFIG)
        if ml_result:
            logger.info("Dataset ML creado exitosamente")
        else:
            logger.warning("No se pudo crear el dataset ML")
    
    # Guardar metadata
    meta = {
        'timestamp': start,
        'finished': datetime.now(timezone.utc).isoformat(),
        'records': len(df_prepared),
        'config': asdict(CONFIG),
        'analysis_summary': {
            'chi2_p': analysis.get('chi2_uniform', {}).get('p'),
            'normality_p': analysis.get('normality', {}).get('p'),
            'entropia_ratio': analysis.get('entropia', {}).get('ratio'),
            'uniformidad_general': analysis.get('chi2_uniform', {}).get('uniforme', False)
        },
        'ml_created': ml_result is not None,
        'ml_target': CONFIG.ml_target if ml_result else None
    }
    
    _save_json(meta, Path(CONFIG.output_dir) / 'run_metadata_completo.json')
    logger.info("Run metadata completo guardado")
    
    # Mostrar reporte ejecutivo
    reporte = analysis.get('reporte_ejecutivo', {})
    print("\n" + "=" * 60)
    print("REPORTE EJECUTIVO - SUPER GANA CON ML")
    print("=" * 60)
    print(f"Muestras analizadas: {reporte.get('muestras', 0):,}")
    print(f"Fecha análisis: {reporte.get('fecha_analisis', 'N/A')}")
    
    print("\nCONCLUSIONES PRINCIPALES:")
    for conclusion in reporte.get('conclusiones_principales', []):
        print(f"  • {conclusion}")
    
    if reporte.get('alertas'):
        print("\n⚠️  ALERTAS:")
        for alerta in reporte.get('alertas', []):
            print(f"  • {alerta}")
    
    if reporte.get('recomendaciones'):
        print("\n💡 RECOMENDACIONES:")
        for rec in reporte.get('recomendaciones', []):
            print(f"  • {rec}")
    
    if ml_result:
        print(f"\n📊 DATASET ML CREADO:")
        print(f"  • Target: {CONFIG.ml_target}")
        print(f"  • Features: {len(ml_result.get('features', []))}")
        print(f"  • Train size: {ml_result.get('metadata', {}).get('train_size', 0)}")
        print(f"  • Test size: {ml_result.get('metadata', {}).get('test_size', 0)}")
        print(f"  • Exportado a: {CONFIG.output_dir}")
    
    print("\n" + "=" * 60)
    logger.info("Análisis completado exitosamente")

if __name__ == "__main__":
    main()