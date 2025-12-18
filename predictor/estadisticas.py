# estadisticas.py — Pruebas estadísticas para Super y Triple Gana

import pandas as pd
import sqlite3
from scipy.stats import chi2_contingency

def cargar_datos(juego):
    conn = sqlite3.connect(f'data/{juego}.db')
    df = pd.read_sql_query("SELECT * FROM resultados", conn)
    conn.close()
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['hora_num'] = df['hora'].str.extract(r'(\d+)').astype(int)
    df['hora_num'] = df.apply(lambda row: row['hora_num'] + 12 if 'pm' in row['hora'] and row['hora_num'] != 12 else row['hora_num'], axis=1)
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce').dropna().astype(int)
    df['paridad'] = df['numero'] % 2
    df['rango'] = pd.cut(df['numero'], bins=[0, 2500, 5000, 7500, 10000], labels=['0–2500', '2501–5000', '5001–7500', '7501–10000'])
    return df

def prueba_chi2_signo_vs_paridad(df):
    tabla = pd.crosstab(df['signo'], df['paridad'])
    chi2, p, dof, expected = chi2_contingency(tabla)
    print("Chi-cuadrado signo vs paridad:")
    print(f"Chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}")
    print("Tabla observada:\n", tabla)

def prueba_chi2_hora_vs_rango(df):
    tabla = pd.crosstab(df['hora_num'], df['rango'])
    chi2, p, dof, expected = chi2_contingency(tabla)
    print("Chi-cuadrado hora vs rango:")
    print(f"Chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}")
    print("Tabla observada:\n", tabla)

def analizar_estadisticas():
    df_super = cargar_datos('supergana')
    df_triple = cargar_datos('triplegana')
    df = pd.concat([df_super, df_triple], ignore_index=True)

    prueba_chi2_signo_vs_paridad(df)
    prueba_chi2_hora_vs_rango(df)

if __name__ == "__main__":
    analizar_estadisticas()
