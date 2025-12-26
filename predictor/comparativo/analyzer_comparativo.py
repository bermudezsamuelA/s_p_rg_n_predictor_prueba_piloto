"""
analyzer_comparativo_avanzado.py - Análisis comparativo avanzado entre loterías
Mejoras: Manejo de errores, análisis estadístico profundo, más visualizaciones, reporte ejecutivo detallado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timezone
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

# -------------------------
# CONFIGURACIÓN
# -------------------------
class ConfigComparativo:
    """Configuración para análisis comparativo"""
    
    def __init__(self):
        # Definir paths relativos
        self.paths = {
            'supergana': {
                'name': 'Super Gana',
                'color': '#1f77b4',  # Azul
                'datos': 'data/analysis/supergana_preparado.csv',
                'reporte': 'data/analysis/supergana_tests_completos.json',
                'stats': 'data/analysis/supergana_tests_summary.csv'
            },
            'triplegana': {
                'name': 'Triple Gana', 
                'color': '#ff7f0e',  # Naranja
                'datos': 'data/analysis_triplegana/triplegana_preparado.csv',
                'reporte': 'data/analysis_triplegana/reporte_triplegana.json',
                'stats': 'data/analysis_triplegana/triplegana_stats.csv'
            }
        }
        
        self.output_dir = 'data/comparativo_avanzado'
        self.figures_dir = f'{self.output_dir}/figuras'
        
        # Crear directorios si no existen
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        
        # Configuración de análisis
        self.uniformity_alpha = 0.05
        self.hot_cold_threshold = 10.0  # % para dígitos calientes/fríos
        self.display_top_n = 15
        
        # Configurar estilo de gráficas
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.output_dir) / 'comparativo.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

# -------------------------
# ANALIZADOR COMPARATIVO AVANZADO
# -------------------------
class ComparadorAvanzado:
    """Clase principal para análisis comparativo avanzado"""
    
    def __init__(self, config: Optional[ConfigComparativo] = None):
        self.config = config or ConfigComparativo()
        self.datos = {}
        self.reportes = {}
        self.estadisticas = {}
        self.resultados = {}
        
    def cargar_datos(self) -> Dict[str, bool]:
        """Carga datos de todos los juegos configurados con manejo de errores robusto"""
        self.config.logger.info("📂 Iniciando carga de datos comparativos...")
        
        resultados_carga = {}
        
        for juego_id, paths in self.config.paths.items():
            try:
                self.config.logger.info(f"  Cargando {paths['name']}...")
                
                # Cargar datos preparados
                if Path(paths['datos']).exists():
                    df = pd.read_csv(paths['datos'])
                    self.datos[juego_id] = df
                    self.config.logger.info(f"    ✓ Datos: {len(df)} registros")
                else:
                    self.config.logger.warning(f"    ✗ Archivo no encontrado: {paths['datos']}")
                    continue
                
                # Cargar reporte JSON si existe
                if Path(paths['reporte']).exists():
                    with open(paths['reporte'], 'r', encoding='utf-8') as f:
                        self.reportes[juego_id] = json.load(f)
                    self.config.logger.info(f"    ✓ Reporte cargado")
                
                # Cargar estadísticas si existen
                if Path(paths['stats']).exists():
                    stats_df = pd.read_csv(paths['stats'])
                    self.estadisticas[juego_id] = stats_df
                
                resultados_carga[juego_id] = True
                
            except Exception as e:
                self.config.logger.error(f"Error cargando {paths['name']}: {str(e)}")
                resultados_carga[juego_id] = False
        
        # Verificar que haya al menos 2 juegos cargados
        juegos_cargados = sum(resultados_carga.values())
        if juegos_cargados < 2:
            raise ValueError(f"Solo {juegos_cargados} juego(s) cargado(s). Se necesitan al menos 2 para comparar.")
        
        self.config.logger.info("✅ Carga de datos completada")
        return resultados_carga
    
    def calcular_estadisticas_comparativas(self) -> pd.DataFrame:
        """Calcula estadísticas comparativas detalladas"""
        self.config.logger.info("📊 Calculando estadísticas comparativas...")
        
        stats_list = []
        
        for juego_id, df in self.datos.items():
            if 'numero_int' not in df.columns:
                df['numero_int'] = pd.to_numeric(df.get('numero', 0), errors='coerce')
            
            stats = {
                'Juego': self.config.paths[juego_id]['name'],
                'Muestras': len(df),
                'Min': df['numero_int'].min(),
                'Max': df['numero_int'].max(),
                'Rango': df['numero_int'].max() - df['numero_int'].min(),
                'Media': df['numero_int'].mean(),
                'Mediana': df['numero_int'].median(),
                'Moda': df['numero_int'].mode().iloc[0] if not df['numero_int'].mode().empty else None,
                'Std': df['numero_int'].std(),
                'Varianza': df['numero_int'].var(),
                'Skewness': df['numero_int'].skew(),
                'Kurtosis': df['numero_int'].kurtosis(),
                'Q1': df['numero_int'].quantile(0.25),
                'Q3': df['numero_int'].quantile(0.75),
                'IQR': df['numero_int'].quantile(0.75) - df['numero_int'].quantile(0.25),
                'CV': (df['numero_int'].std() / df['numero_int'].mean()) * 100 if df['numero_int'].mean() != 0 else None
            }
            
            # Calcular uniformidad (test KS)
            try:
                ks_stat, ks_p = stats.kstest(
                    (df['numero_int'] - df['numero_int'].min()) / 
                    (df['numero_int'].max() - df['numero_int'].min()),
                    'uniform'
                )
                stats['KS_Uniform_p'] = ks_p
                stats['Es_Uniforme'] = ks_p > self.config.uniformity_alpha
            except:
                stats['KS_Uniform_p'] = None
                stats['Es_Uniforme'] = None
            
            stats_list.append(stats)
        
        df_stats = pd.DataFrame(stats_list)
        df_stats.set_index('Juego', inplace=True)
        
        # Calcular diferencias relativas
        if len(df_stats) >= 2:
            base = df_stats.iloc[0]
            for idx in range(1, len(df_stats)):
                juego = df_stats.index[idx]
                for col in ['Media', 'Std', 'Skewness', 'Kurtosis']:
                    diff_col = f'Diff_{col}_%'
                    if col in base and col in df_stats.loc[juego]:
                        if base[col] != 0:
                            df_stats.loc[juego, diff_col] = (
                                (df_stats.loc[juego, col] - base[col]) / base[col] * 100
                            )
        
        # Guardar estadísticas
        df_stats.to_csv(Path(self.config.output_dir) / 'estadisticas_comparativas_detalladas.csv')
        self.config.logger.info(f"✅ Estadísticas guardadas: {len(df_stats)} juegos comparados")
        
        self.resultados['estadisticas'] = df_stats
        return df_stats
    
    def analizar_digitos_por_posicion(self) -> Dict[str, Any]:
        """Análisis comparativo de dígitos por posición"""
        self.config.logger.info("🔢 Analizando dígitos por posición...")
        
        analisis_digitos = {}
        
        for juego_id, df in self.datos.items():
            juego_nombre = self.config.paths[juego_id]['name']
            analisis_digitos[juego_nombre] = {}
            
            # Verificar columnas de dígitos
            posiciones = ['primer_digito', 'segundo_digito', 'tercer_digito', 'cuarto_digito']
            posiciones_disponibles = [p for p in posiciones if p in df.columns]
            
            for pos in posiciones_disponibles:
                if pos in df.columns:
                    frecuencias = df[pos].value_counts().sort_index()
                    porcentajes = (frecuencias / len(df) * 100).round(2)
                    
                    # Calcular desviaciones
                    esperado = 10.0  # 10% para cada dígito 0-9
                    desviaciones = ((porcentajes - esperado) / esperado * 100).round(2)
                    
                    # Identificar dígitos destacados
                    calientes = desviaciones[desviaciones > self.config.hot_cold_threshold].index.tolist()
                    frios = desviaciones[desviaciones < -self.config.hot_cold_threshold].index.tolist()
                    
                    # Test chi-cuadrado para uniformidad
                    chi2, p = stats.chisquare(frecuencias)
                    
                    analisis_digitos[juego_nombre][pos] = {
                        'frecuencias': frecuencias.to_dict(),
                        'porcentajes': porcentajes.to_dict(),
                        'desviaciones': desviaciones.to_dict(),
                        'calientes': calientes,
                        'frios': frios,
                        'chi2': float(chi2),
                        'p_value': float(p),
                        'uniforme': p > self.config.uniformity_alpha,
                        'max_desviacion': float(desviaciones.abs().max())
                    }
        
        # Guardar análisis
        with open(Path(self.config.output_dir) / 'analisis_digitos_comparativo.json', 'w') as f:
            json.dump(analisis_digitos, f, indent=2, default=str)
        
        self.resultados['digitos'] = analisis_digitos
        return analisis_digitos
    
    def comparar_uniformidad_tests(self) -> pd.DataFrame:
        """Compara resultados de tests de uniformidad entre juegos"""
        self.config.logger.info("🎲 Comparando tests de uniformidad...")
        
        tests_data = []
        
        for juego_id, reporte in self.reportes.items():
            juego_nombre = self.config.paths[juego_id]['name']
            
            # Extraer resultados de tests del reporte
            test_info = {
                'Juego': juego_nombre,
                'Chi2_p': reporte.get('chi2_uniform', {}).get('p', None),
                'Chi2_Uniforme': reporte.get('chi2_uniform', {}).get('uniforme', None),
                'Entropia_Ratio': reporte.get('entropia', {}).get('ratio', None),
                'Entropia_Alta': reporte.get('entropia', {}).get('alta_entropia', None),
                'Runs_p': reporte.get('runs_test', {}).get('p', None) if reporte.get('runs_test') else None,
                'Runs_Aleatorio': reporte.get('runs_test', {}).get('aleatorio', None) if reporte.get('runs_test') else None,
                'LjungBox_Indep': reporte.get('ljung_box', {}).get('independiente', None) if reporte.get('ljung_box') else None
            }
            
            tests_data.append(test_info)
        
        df_tests = pd.DataFrame(tests_data)
        df_tests.set_index('Juego', inplace=True)
        
        # Guardar
        df_tests.to_csv(Path(self.config.output_dir) / 'tests_uniformidad_comparados.csv')
        
        self.resultados['tests_uniformidad'] = df_tests
        return df_tests
    
    def crear_visualizaciones_avanzadas(self):
        """Crea visualizaciones comparativas avanzadas"""
        self.config.logger.info("📈 Creando visualizaciones avanzadas...")
        
        # 1. Dashboard comparativo principal
        self._crear_dashboard_comparativo()
        
        # 2. Heatmap de correlaciones cruzadas
        self._crear_heatmap_correlaciones()
        
        # 3. Análisis temporal comparativo
        self._crear_analisis_temporal()
        
        # 4. Distribución acumulada comparativa
        self._crear_distribucion_acumulada()
        
        # 5. Radar chart de métricas clave
        self._crear_radar_chart()
        
        self.config.logger.info("✅ Visualizaciones creadas")
    
    def _crear_dashboard_comparativo(self):
        """Dashboard comparativo con múltiples gráficas"""
        fig = plt.figure(figsize=(20, 16))
        
        juegos = list(self.datos.keys())
        n_juegos = len(juegos)
        
        # 1. Distribución de números (histogramas superpuestos)
        ax1 = plt.subplot(3, 3, 1)
        for idx, juego_id in enumerate(juegos):
            df = self.datos[juego_id]
            color = self.config.paths[juego_id]['color']
            label = self.config.paths[juego_id]['name']
            
            if 'numero_int' in df.columns:
                sns.histplot(df['numero_int'], bins=50, kde=True, 
                            alpha=0.5, color=color, label=label, ax=ax1)
        
        ax1.set_title('Distribución de Números - Comparativa', fontweight='bold')
        ax1.set_xlabel('Número')
        ax1.set_ylabel('Densidad')
        ax1.legend()
        
        # 2. Boxplot comparativo
        ax2 = plt.subplot(3, 3, 2)
        box_data = []
        labels = []
        colors = []
        
        for juego_id in juegos:
            df = self.datos[juego_id]
            if 'numero_int' in df.columns:
                box_data.append(df['numero_int'].dropna().values)
                labels.append(self.config.paths[juego_id]['name'])
                colors.append(self.config.paths[juego_id]['color'])
        
        box = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Boxplot Comparativo', fontweight='bold')
        ax2.set_ylabel('Número')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Último dígito comparado
        ax3 = plt.subplot(3, 3, 3)
        x = np.arange(10)
        width = 0.8 / n_juegos
        
        for idx, juego_id in enumerate(juegos):
            df = self.datos[juego_id]
            color = self.config.paths[juego_id]['color']
            label = self.config.paths[juego_id]['name']
            
            if 'ultimo_digito' in df.columns:
                frec = df['ultimo_digito'].value_counts().sort_index()
                ax3.bar(x + idx*width - width*(n_juegos-1)/2, 
                       frec.values, width, label=label, color=color, alpha=0.8)
        
        ax3.axhline(y=len(self.datos[juegos[0]])/10, color='red', linestyle='--', 
                   label='Esperado (10%)', alpha=0.7)
        ax3.set_title('Último Dígito - Distribución', fontweight='bold')
        ax3.set_xlabel('Dígito')
        ax3.set_ylabel('Frecuencia')
        ax3.set_xticks(x)
        ax3.legend()
        
        # 4. Heatmap de dígitos por posición (solo si hay 2 juegos)
        if n_juegos == 2:
            ax4 = plt.subplot(3, 3, 4)
            self._crear_heatmap_digitos(ax4)
        
        # 5. Media móvil comparativa (si hay fecha)
        ax5 = plt.subplot(3, 3, 5)
        for juego_id in juegos:
            df = self.datos[juego_id]
            color = self.config.paths[juego_id]['color']
            label = self.config.paths[juego_id]['name']
            
            if 'fecha' in df.columns and 'numero_int' in df.columns:
                df_sorted = df.sort_values('fecha').reset_index(drop=True)
                if len(df_sorted) > 30:
                    df_sorted['media_movil'] = df_sorted['numero_int'].rolling(30).mean()
                    ax5.plot(df_sorted.index, df_sorted['media_movil'], 
                            label=label, color=color, linewidth=2, alpha=0.8)
        
        ax5.set_title('Media Móvil 30 días', fontweight='bold')
        ax5.set_xlabel('Índice')
        ax5.set_ylabel('Número (media móvil)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Paridad comparativa
        ax6 = plt.subplot(3, 3, 6)
        paridad_data = []
        paridad_labels = []
        
        for juego_id in juegos:
            df = self.datos[juego_id]
            if 'paridad' in df.columns:
                par_counts = df['paridad'].value_counts()
                if len(par_counts) == 2:
                    paridad_data.append([
                        par_counts.get(0, 0) / len(df) * 100,
                        par_counts.get(1, 0) / len(df) * 100
                    ])
                    paridad_labels.append(self.config.paths[juego_id]['name'])
        
        if paridad_data:
            x = np.arange(len(paridad_data))
            width = 0.35
            
            for idx in range(len(paridad_data)):
                ax6.bar(x[idx] - width/2, paridad_data[idx][0], width, 
                       label='Pares' if idx == 0 else '', color='skyblue')
                ax6.bar(x[idx] + width/2, paridad_data[idx][1], width, 
                       label='Impares' if idx == 0 else '', color='lightcoral')
            
            ax6.set_title('Distribución Par/Impar', fontweight='bold')
            ax6.set_xlabel('Juego')
            ax6.set_ylabel('Porcentaje (%)')
            ax6.set_xticks(x)
            ax6.set_xticklabels(paridad_labels)
            ax6.legend(['Pares', 'Impares'])
        
        # 7. Tests estadísticos comparados
        ax7 = plt.subplot(3, 3, 7)
        if 'tests_uniformidad' in self.resultados:
            df_tests = self.resultados['tests_uniformidad']
            
            # Preparar datos para gráfico de barras
            metrics = ['Chi2_Uniforme', 'Entropia_Alta', 'Runs_Aleatorio']
            metric_labels = ['Chi² Uniforme', 'Alta Entropía', 'Runs Aleatorio']
            
            x = np.arange(len(df_tests))
            width = 0.25
            
            for idx, metric in enumerate(metrics):
                if metric in df_tests.columns:
                    values = df_tests[metric].apply(lambda x: 1 if x else 0).values
                    ax7.bar(x + idx*width - width, values, width, 
                           label=metric_labels[idx], alpha=0.7)
            
            ax7.set_title('Tests de Aleatoriedad', fontweight='bold')
            ax7.set_xlabel('Juego')
            ax7.set_ylabel('Resultado (1=Positivo)')
            ax7.set_xticks(x)
            ax7.set_xticklabels(df_tests.index)
            ax7.set_ylim(0, 1.5)
            ax7.legend()
        
        # 8. Top números comparados
        ax8 = plt.subplot(3, 3, 8)
        for idx, juego_id in enumerate(juegos):
            df = self.datos[juego_id]
            if 'numero_int' in df.columns:
                top_n = df['numero_int'].value_counts().head(self.config.display_top_n)
                
                if idx == 0:
                    ax8.barh(range(len(top_n)), top_n.values, 
                            color=self.config.paths[juego_id]['color'], alpha=0.7,
                            label=self.config.paths[juego_id]['name'])
                    ax8.set_yticks(range(len(top_n)))
                    ax8.set_yticklabels([str(n) for n in top_n.index])
                else:
                    # Para comparación lado a lado
                    pass
        
        ax8.set_title(f'Top {self.config.display_top_n} Números', fontweight='bold')
        ax8.set_xlabel('Frecuencia')
        ax8.invert_yaxis()
        ax8.legend()
        
        # 9. Resumen comparativo
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        if 'estadisticas' in self.resultados:
            df_stats = self.resultados['estadisticas']
            
            summary_text = "RESUMEN COMPARATIVO\n" + "="*30 + "\n\n"
            
            for juego in df_stats.index:
                summary_text += f"{juego}:\n"
                summary_text += f"  Muestras: {int(df_stats.loc[juego, 'Muestras']):,}\n"
                summary_text += f"  Media: {df_stats.loc[juego, 'Media']:.0f}\n"
                summary_text += f"  Std: {df_stats.loc[juego, 'Std']:.0f}\n"
                
                if 'Es_Uniforme' in df_stats.columns and not pd.isna(df_stats.loc[juego, 'Es_Uniforme']):
                    uniforme = "✓" if df_stats.loc[juego, 'Es_Uniforme'] else "✗"
                    summary_text += f"  Uniforme: {uniforme}\n"
                
                summary_text += "\n"
            
            ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('DASHBOARD COMPARATIVO AVANZADO', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Guardar
        plt.savefig(Path(self.config.figures_dir) / 'dashboard_comparativo_avanzado.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _crear_heatmap_digitos(self, ax):
        """Heatmap para comparar dígitos entre dos juegos"""
        if len(self.datos) != 2:
            return
        
        juegos = list(self.datos.keys())
        juego1, juego2 = juegos[0], juegos[1]
        
        # Obtener frecuencias de último dígito
        if 'ultimo_digito' in self.datos[juego1].columns and 'ultimo_digito' in self.datos[juego2].columns:
            freq1 = self.datos[juego1]['ultimo_digito'].value_counts().sort_index()
            freq2 = self.datos[juego2]['ultimo_digito'].value_counts().sort_index()
            
            # Calcular diferencias porcentuales
            total1 = len(self.datos[juego1])
            total2 = len(self.datos[juego2])
            
            pct1 = (freq1 / total1 * 100).reindex(range(10), fill_value=0)
            pct2 = (freq2 / total2 * 100).reindex(range(10), fill_value=0)
            
            diferencias = (pct1 - pct2).values.reshape(1, 10)
            
            # Crear heatmap
            im = ax.imshow(diferencias, cmap='RdBu_r', aspect='auto', 
                          vmin=-20, vmax=20)
            
            ax.set_title('Diferencia % en Último Dígito', fontweight='bold')
            ax.set_xticks(range(10))
            ax.set_xticklabels(range(10))
            ax.set_yticks([0])
            ax.set_yticklabels(['Diferencia'])
            
            # Añadir valores
            for i in range(10):
                valor = diferencias[0, i]
                color = 'white' if abs(valor) > 10 else 'black'
                ax.text(i, 0, f'{valor:+.1f}%', ha='center', va='center', 
                       color=color, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Diferencia %')
    
    def _crear_heatmap_correlaciones(self):
        """Heatmap de correlaciones cruzadas entre juegos"""
        if len(self.datos) < 2:
            return
        
        # Preparar datos para correlación
        correlaciones = {}
        
        for juego_id, df in self.datos.items():
            if 'numero_int' in df.columns:
                # Crear serie temporal si hay fecha
                if 'fecha' in df.columns:
                    df_sorted = df.sort_values('fecha')
                    correlaciones[juego_id] = df_sorted['numero_int'].reset_index(drop=True)
                else:
                    correlaciones[juego_id] = df['numero_int'].reset_index(drop=True)
        
        # Crear DataFrame de correlaciones
        max_len = max(len(s) for s in correlaciones.values())
        corr_df = pd.DataFrame()
        
        for juego_id, serie in correlaciones.items():
            juego_nombre = self.config.paths[juego_id]['name']
            # Alinear series por longitud
            aligned = serie.reindex(range(max_len))
            corr_df[juego_nombre] = aligned
        
        # Calcular matriz de correlación
        corr_matrix = corr_df.corr()
        
        # Crear heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Matriz de Correlación entre Juegos', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(Path(self.config.figures_dir) / 'heatmap_correlaciones.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _crear_analisis_temporal(self):
        """Análisis temporal comparativo"""
        juegos_con_fecha = []
        
        for juego_id, df in self.datos.items():
            if 'fecha' in df.columns and 'numero_int' in df.columns:
                juegos_con_fecha.append(juego_id)
        
        if len(juegos_con_fecha) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, juego_id in enumerate(juegos_con_fecha[:4]):  # Máximo 4 juegos
            if idx >= len(axes):
                break
                
            df = self.datos[juego_id]
            df['fecha_dt'] = pd.to_datetime(df['fecha'])
            df_sorted = df.sort_values('fecha_dt')
            
            # Gráfico de serie temporal
            axes[idx].plot(df_sorted['fecha_dt'], df_sorted['numero_int'], 
                          'o', markersize=2, alpha=0.5,
                          color=self.config.paths[juego_id]['color'])
            
            # Media móvil
            if len(df_sorted) > 30:
                df_sorted['media_movil'] = df_sorted['numero_int'].rolling(30).mean()
                axes[idx].plot(df_sorted['fecha_dt'], df_sorted['media_movil'], 
                              linewidth=2, color='red', alpha=0.8)
            
            axes[idx].set_title(f"{self.config.paths[juego_id]['name']} - Evolución", 
                               fontweight='bold')
            axes[idx].set_xlabel('Fecha')
            axes[idx].set_ylabel('Número')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('ANÁLISIS TEMPORAL COMPARATIVO', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(self.config.figures_dir) / 'analisis_temporal.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _crear_distribucion_acumulada(self):
        """Distribución acumulada comparativa"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for juego_id, df in self.datos.items():
            if 'numero_int' in df.columns:
                data = df['numero_int'].dropna().values
                data_sorted = np.sort(data)
                y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                
                ax.plot(data_sorted, y, linewidth=2, alpha=0.8,
                       color=self.config.paths[juego_id]['color'],
                       label=self.config.paths[juego_id]['name'])
        
        ax.set_title('Distribución Acumulada Comparativa', fontweight='bold')
        ax.set_xlabel('Número')
        ax.set_ylabel('Probabilidad Acumulada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.figures_dir) / 'distribucion_acumulada.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _crear_radar_chart(self):
        """Radar chart de métricas clave"""
        if 'estadisticas' not in self.resultados:
            return
        
        df_stats = self.resultados['estadisticas']
        
        # Seleccionar métricas para radar chart
        metrics = ['Media', 'Std', 'Skewness', 'Kurtosis', 'CV']
        metric_labels = ['Media', 'Desv. Std', 'Sesgo', 'Curtosis', 'Coef. Var']
        
        # Normalizar valores
        radar_data = []
        juegos = df_stats.index.tolist()
        
        for juego in juegos:
            juego_data = []
            for metric in metrics:
                if metric in df_stats.columns and not pd.isna(df_stats.loc[juego, metric]):
                    valor = df_stats.loc[juego, metric]
                    # Normalizar entre 0 y 1 (simplificado)
                    juego_data.append(valor)
                else:
                    juego_data.append(0)
            radar_data.append(juego_data)
        
        if not radar_data:
            return
        
        # Crear radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for idx, juego in enumerate(juegos):
            valores = radar_data[idx] + radar_data[idx][:1]  # Cerrar el círculo
            color = self.config.colors[idx % len(self.config.colors)]
            
            ax.plot(angles, valores, 'o-', linewidth=2, color=color, 
                   label=juego, alpha=0.7)
            ax.fill(angles, valores, color=color, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_title('Radar Chart - Métricas Comparativas', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.figures_dir) / 'radar_chart_metricas.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generar_reporte_ejecutivo(self):
        """Genera reporte ejecutivo comparativo detallado"""
        self.config.logger.info("📋 Generando reporte ejecutivo...")
        
        reporte = {
            'metadata': {
                'generado_en': datetime.now(timezone.utc).isoformat(),
                'juegos_analizados': [self.config.paths[jid]['name'] for jid in self.datos.keys()],
                'total_muestras': sum(len(df) for df in self.datos.values()),
                'configuracion': {
                    'alpha_uniformidad': self.config.uniformity_alpha,
                    'threshold_caliente_frio': self.config.hot_cold_threshold
                }
            },
            'resumen_estadistico': {},
            'comparativa_aleatoriedad': {},
            'hallazgos_clave': [],
            'recomendaciones': []
        }
        
        # Resumen estadístico
        if 'estadisticas' in self.resultados:
            df_stats = self.resultados['estadisticas']
            reporte['resumen_estadistico'] = df_stats.to_dict(orient='index')
        
        # Comparativa de aleatoriedad
        if 'tests_uniformidad' in self.resultados:
            df_tests = self.resultados['tests_uniformidad']
            reporte['comparativa_aleatoriedad'] = df_tests.to_dict(orient='index')
        
        # Hallazgos clave
        self._identificar_hallazgos_clave(reporte)
        
        # Recomendaciones
        self._generar_recomendaciones(reporte)
        
        # Guardar reporte JSON
        with open(Path(self.config.output_dir) / 'reporte_comparativo_completo.json', 'w') as f:
            json.dump(reporte, f, indent=2, default=str)
        
        # Generar reporte en texto
        self._generar_reporte_texto(reporte)
        
        self.config.logger.info("✅ Reporte ejecutivo generado")
        return reporte
    
    def _identificar_hallazgos_clave(self, reporte: Dict):
        """Identifica hallazgos clave del análisis comparativo"""
        if 'estadisticas' not in self.resultados:
            return
        
        df_stats = self.resultados['estadisticas']
        juegos = df_stats.index.tolist()
        
        if len(juegos) < 2:
            return
        
        # Hallazgo 1: Comparación de uniformidad
        if 'Es_Uniforme' in df_stats.columns:
            uniformes = df_stats['Es_Uniforme'].dropna()
            if len(uniformes) >= 2:
                if uniformes.all():
                    reporte['hallazgos_clave'].append("Todos los juegos muestran distribución uniforme (aleatoria)")
                elif not uniformes.any():
                    reporte['hallazgos_clave'].append("Todos los juegos muestran posible sesgo (no uniformes)")
                else:
                    uniforme_true = uniformes[uniformes == True].index.tolist()
                    uniforme_false = uniformes[uniformes == False].index.tolist()
                    if uniforme_true:
                        reporte['hallazgos_clave'].append(f"Juegos uniformes: {', '.join(uniforme_true)}")
                    if uniforme_false:
                        reporte['hallazgos_clave'].append(f"Juegos con posible sesgo: {', '.join(uniforme_false)}")
        
        # Hallazgo 2: Comparación de dispersión
        if 'Std' in df_stats.columns:
            std_vals = df_stats['Std'].dropna()
            if len(std_vals) >= 2:
                max_std = std_vals.idxmax()
                min_std = std_vals.idxmin()
                ratio = std_vals.max() / std_vals.min()
                
                if ratio > 1.5:
                    reporte['hallazgos_clave'].append(
                        f"{max_std} tiene mayor dispersión (std={std_vals[max_std]:.0f}) "
                        f"vs {min_std} (std={std_vals[min_std]:.0f})"
                    )
        
        # Hallazgo 3: Dígitos calientes/fríos coincidentes
        if 'digitos' in self.resultados:
            analisis_digitos = self.resultados['digitos']
            juegos_digitos = list(analisis_digitos.keys())
            
            if len(juegos_digitos) >= 2:
                # Buscar dígitos calientes coincidentes
                calientes_comunes = set()
                frios_comunes = set()
                
                for juego in juegos_digitos:
                    if 'cuarto_digito' in analisis_digitos[juego]:
                        calientes = set(analisis_digitos[juego]['cuarto_digito']['calientes'])
                        frios = set(analisis_digitos[juego]['cuarto_digito']['frios'])
                        
                        if not calientes_comunes:
                            calientes_comunes = calientes
                            frios_comunes = frios
                        else:
                            calientes_comunes &= calientes
                            frios_comunes &= frios
                
                if calientes_comunes:
                    reporte['hallazgos_clave'].append(
                        f"Dígitos calientes comunes: {sorted(calientes_comunes)}"
                    )
                if frios_comunes:
                    reporte['hallazgos_clave'].append(
                        f"Dígitos fríos comunes: {sorted(frios_comunes)}"
                    )
    
    def _generar_recomendaciones(self, reporte: Dict):
        """Genera recomendaciones basadas en el análisis"""
        
        # Recomendación general
        reporte['recomendaciones'].append(
            "Monitorear periódicamente la distribución para detectar cambios en los patrones"
        )
        
        # Recomendaciones basadas en uniformidad
        if 'estadisticas' in self.resultados:
            df_stats = self.resultados['estadisticas']
            
            if 'Es_Uniforme' in df_stats.columns:
                uniformes = df_stats['Es_Uniforme'].dropna()
                
                if not uniformes.all():
                    reporte['recomendaciones'].append(
                        "Considerar estrategias basadas en dígitos/pautas identificadas para juegos no uniformes"
                    )
                else:
                    reporte['recomendaciones'].append(
                        "Todos los juegos son aleatorios - cualquier estrategia tiene probabilidad similar"
                    )
        
        # Recomendación basada en dígitos
        if 'digitos' in self.resultados:
            reporte['recomendaciones'].append(
                "Utilizar análisis de dígitos calientes/fríos para identificar oportunidades de juego"
            )
        
        # Recomendación para ML
        reporte['recomendaciones'].append(
            "Considerar modelos predictivos separados para cada juego debido a posibles diferencias en patrones"
        )
    
    def _generar_reporte_texto(self, reporte: Dict):
        """Genera reporte ejecutivo en texto plano"""
        with open(Path(self.config.output_dir) / 'reporte_ejecutivo.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE EJECUTIVO - ANÁLISIS COMPARATIVO AVANZADO\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("📊 RESUMEN GENERAL\n")
            f.write("-" * 40 + "\n")
            f.write(f"Fecha análisis: {reporte['metadata']['generado_en']}\n")
            f.write(f"Juegos analizados: {', '.join(reporte['metadata']['juegos_analizados'])}\n")
            f.write(f"Total muestras: {reporte['metadata']['total_muestras']:,}\n\n")
            
            f.write("🎯 HALLAZGOS CLAVE\n")
            f.write("-" * 40 + "\n")
            if reporte['hallazgos_clave']:
                for hallazgo in reporte['hallazgos_clave']:
                    f.write(f"• {hallazgo}\n")
            else:
                f.write("No se identificaron hallazgos significativos\n")
            f.write("\n")
            
            f.write("💡 RECOMENDACIONES\n")
            f.write("-" * 40 + "\n")
            if reporte['recomendaciones']:
                for i, recomendacion in enumerate(reporte['recomendaciones'], 1):
                    f.write(f"{i}. {recomendacion}\n")
            f.write("\n")
            
            f.write("📈 ESTADÍSTICAS COMPARATIVAS\n")
            f.write("-" * 40 + "\n")
            if 'resumen_estadistico' in reporte and reporte['resumen_estadistico']:
                for juego, stats in reporte['resumen_estadistico'].items():
                    f.write(f"\n{juego}:\n")
                    f.write(f"  Muestras: {int(stats.get('Muestras', 0)):,}\n")
                    f.write(f"  Media: {stats.get('Media', 'N/A'):.0f}\n")
                    f.write(f"  Desv. Std: {stats.get('Std', 'N/A'):.0f}\n")
                    f.write(f"  Rango: {stats.get('Min', 'N/A'):.0f} - {stats.get('Max', 'N/A'):.0f}\n")
                    if 'Es_Uniforme' in stats and stats['Es_Uniforme'] is not None:
                        uniforme = "✓ SÍ" if stats['Es_Uniforme'] else "✗ NO"
                        f.write(f"  Distribución uniforme: {uniforme}\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("ANÁLISIS COMPLETADO - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 70 + "\n")
        
        # También mostrar en consola
        print("\n" + "=" * 70)
        print("REPORTE EJECUTIVO - RESUMEN")
        print("=" * 70)
        
        print("\n🎯 HALLAZGOS PRINCIPALES:")
        for hallazgo in reporte.get('hallazgos_clave', [])[:5]:  # Mostrar solo los 5 principales
            print(f"  • {hallazgo}")
        
        print("\n💡 RECOMENDACIÓN PRINCIPAL:")
        if reporte.get('recomendaciones'):
            print(f"  {reporte['recomendaciones'][0]}")
        
        print("\n" + "=" * 70)
    
    def ejecutar_analisis_completo(self):
        """Ejecuta todo el pipeline de análisis comparativo"""
        self.config.logger.info("🚀 Iniciando análisis comparativo completo...")
        
        try:
            # 1. Cargar datos
            self.cargar_datos()
            
            # 2. Análisis estadístico
            self.calcular_estadisticas_comparativas()
            
            # 3. Análisis de dígitos
            self.analizar_digitos_por_posicion()
            
            # 4. Comparar tests de uniformidad
            self.comparar_uniformidad_tests()
            
            # 5. Visualizaciones
            self.crear_visualizaciones_avanzadas()
            
            # 6. Reporte ejecutivo
            self.generar_reporte_ejecutivo()
            
            self.config.logger.info("✅ Análisis comparativo completado exitosamente")
            print(f"\n📁 Resultados guardados en: {self.config.output_dir}/")
            
        except Exception as e:
            self.config.logger.error(f"Error en análisis comparativo: {str(e)}")
            raise

# -------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------
def main():
    """Función principal para ejecutar el análisis comparativo"""
    print("\n" + "="*70)
    print("ANÁLISIS COMPARATIVO AVANZADO - SUPER GANA vs TRIPLE GANA")
    print("="*70 + "\n")
    
    try:
        # Crear y configurar comparador
        comparador = ComparadorAvanzado()
        
        # Ejecutar análisis completo
        comparador.ejecutar_analisis_completo()
        
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPARATIVO COMPLETADO CON ÉXITO")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())