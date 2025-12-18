# analyzer_triplegana.py — Análisis descriptivo de Triple Gana

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos():
    conn = sqlite3.connect('data/triplegana.db')
    df = pd.read_sql_query("SELECT * FROM resultados", conn)
    conn.close()
    return df

def preparar_datos(df):
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['hora_num'] = df['hora'].str.extract(r'(\d+)').astype(int)
    df['hora_num'] = df.apply(lambda row: row['hora_num'] + 12 if 'pm' in row['hora'] and row['hora_num'] != 12 else row['hora_num'], axis=1)
    df['dia_final'] = df['fecha'].dt.day % 10
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
    df = df.dropna(subset=['numero'])
    df['numero'] = df['numero'].astype(int)
    df['paridad'] = df['numero'] % 2
    df['rango'] = pd.cut(df['numero'], bins=[0, 2500, 5000, 7500, 10000], labels=['0–2500', '2501–5000', '5001–7500', '7501–10000'])
    return df

def graficar_numeros_frecuentes(df):
    plt.figure(figsize=(12, 6))
    df['numero'].value_counts().head(20).sort_index().plot(kind='bar', color='steelblue')
    plt.title('Top 20 números más frecuentes (Triple Gana)')
    plt.xlabel('Número')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def graficar_signos(df):
    plt.figure(figsize=(10, 5))
    df['signo'].value_counts().plot(kind='bar', color='purple')
    plt.title('Distribución de signos zodiacales (Triple Gana)')
    plt.xlabel('Signo')
    plt.ylabel('Cantidad de apariciones')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analizar_triplegana():
    df = cargar_datos()
    df = preparar_datos(df)
    graficar_numeros_frecuentes(df)
    graficar_signos(df)

if __name__ == "__main__":
    analizar_triplegana()
