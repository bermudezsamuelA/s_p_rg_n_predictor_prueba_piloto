# analyzer_comparativo.py — Compara Super Gana y Triple Gana

import pandas as pd
import sqlite3

# Orden astrológico para calcular distancia zodiacal
orden_signos = ['ARI', 'TAU', 'GEM', 'CAN', 'LEO', 'VIR', 'LIB', 'ESC', 'SAG', 'CAP', 'ACU', 'PIC']
signo_map = {signo: i for i, signo in enumerate(orden_signos)}

def cargar_datos(juego):
    db_path = f"data/{juego}.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM resultados", conn)
    conn.close()
    df['juego'] = juego
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['hora_num'] = df['hora'].str.extract(r'(\d+)').astype(int)
    df['hora_num'] = df.apply(lambda row: row['hora_num'] + 12 if 'pm' in row['hora'] and row['hora_num'] != 12 else row['hora_num'], axis=1)
    df['numero'] = pd.to_numeric(df['numero'], errors='coerce')
    df = df.dropna(subset=['numero'])
    df['numero'] = df['numero'].astype(int)
    return df

def comparar_por_hora(df):
    tabla = df.groupby(['juego', 'hora_num'])['numero'].value_counts().unstack(fill_value=0)
    print("Frecuencia de números por hora y juego:\n", tabla.head(10))

def comparar_signos(df):
    signos = df.groupby(['juego', 'hora_num'])['signo'].value_counts().unstack(fill_value=0)
    print("Distribución de signos por hora y juego:\n", signos.head(10))

def detectar_coincidencias(df):
    pivot = df.pivot_table(index=['fecha', 'hora'], columns='juego', values='numero', aggfunc='first')
    coincidencias = pivot[pivot['supergana'] == pivot['triplegana']]
    print(f"Coincidencias exactas de número entre juegos: {len(coincidencias)}")
    print(coincidencias.head())
    coincidencias.to_csv('data/coincidencias.csv')

def comparar_distancia_signos(df):
    pivot = df.pivot_table(index=['fecha', 'hora'], columns='juego', values='signo', aggfunc='first')
    pivot = pivot.dropna()
    pivot['distancia_signo'] = pivot.apply(
        lambda row: signo_map.get(row['triplegana'], -1) - signo_map.get(row['supergana'], -1),
        axis=1
    )
    print("Distancia zodiacal entre signos (triplegana - supergana):\n", pivot['distancia_signo'].value_counts().sort_index())
    return pivot[['distancia_signo']]

def comparar_distancia_numeros(df):
    pivot = df.pivot_table(index=['fecha', 'hora'], columns='juego', values='numero', aggfunc='first')
    pivot = pivot.dropna()
    pivot['distancia_num'] = abs(pivot['supergana'] - pivot['triplegana'])
    print("Resumen de distancia numérica entre juegos:\n", pivot['distancia_num'].describe())
    return pivot[['distancia_num']]

def analizar_comparativo():
    df_super = cargar_datos('supergana')
    df_triple = cargar_datos('triplegana')
    df = pd.concat([df_super, df_triple], ignore_index=True)

    comparar_por_hora(df)
    comparar_signos(df)
    detectar_coincidencias(df)

    dist_signos = comparar_distancia_signos(df)
    dist_numeros = comparar_distancia_numeros(df)

    # Exportar distancias
    distancias = pd.concat([dist_signos, dist_numeros], axis=1)
    distancias.to_csv('data/distancias_comparativas.csv')

if __name__ == "__main__":
    analizar_comparativo()
