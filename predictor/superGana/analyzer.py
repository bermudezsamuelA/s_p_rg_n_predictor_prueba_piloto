import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos():
    conn = sqlite3.connect('data/supergana.db')
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
    df['signo_codificado'] = df['signo'].astype('category').cat.codes
    df['hora_codificada'] = df['hora_num']
    return df

def graficar_numeros_frecuentes(df):
    top = df['numero'].value_counts().head(20)
    print("Top 20 números más frecuentes:\n", top)
    top.sort_index().plot(kind='bar', figsize=(12, 6), color='steelblue')
    plt.title('Top 20 números más frecuentes')
    plt.xlabel('Número')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def graficar_numeros_por_dia_final(df):
    plt.figure(figsize=(10, 6))
    for i in range(10):
        subset = df[df['dia_final'] == i]
        plt.scatter([i]*len(subset), subset['numero'], alpha=0.5)
    plt.title('Distribución de números según último dígito del día')
    plt.xlabel('Último dígito del día')
    plt.ylabel('Número sorteado')
    plt.xticks(range(10))
    plt.tight_layout()
    plt.show()

def graficar_signos(df):
    print("Distribución de signos:\n", df['signo'].value_counts())
    df['signo'].value_counts().plot(kind='bar', figsize=(10, 5), color='purple')
    plt.title('Distribución de signos zodiacales')
    plt.xlabel('Signo')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def graficar_signo_vs_paridad(df):
    tabla = df.groupby(['signo', 'paridad']).size().unstack(fill_value=0)
    print("Signo vs Paridad:\n", tabla)
    tabla.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
    plt.title('Distribución de signos según paridad')
    plt.xlabel('Signo')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    plt.legend(['Impar', 'Par'])
    plt.tight_layout()
    plt.show()

def graficar_signo_vs_rango(df):
    tabla = df.groupby(['signo', 'rango']).size().unstack(fill_value=0)
    print("Signo vs Rango:\n", tabla)
    tabla.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title('Distribución de signos según rango del número')
    plt.xlabel('Signo')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analizar_datos():
    df = cargar_datos()
    df = preparar_datos(df)
    df.to_csv('data/supergana_preparado.csv', index=False)
    graficar_numeros_frecuentes(df)
    graficar_numeros_por_dia_final(df)
    graficar_signos(df)
    graficar_signo_vs_paridad(df)
    graficar_signo_vs_rango(df)

if __name__ == "__main__":
    analizar_datos()
