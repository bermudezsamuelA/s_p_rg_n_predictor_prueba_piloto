# extractor_triplegana.py — Extrae resultados de Triple Gana y los guarda en SQLite

import requests
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from db import guardar_resultado, crear_tabla  # Funciones que gestionan la base de datos

# Construye la URL para una fecha específica
def obtener_html(fecha_str):
    url = f"https://supergana.com.ve/pruebah.php?bt={fecha_str.replace('/', '%2F')}"
    response = requests.get(url)
    return response.text

# Extrae los resultados de Triple Gana (segundo td después del th)
def extraer_resultados(html):
    soup = BeautifulSoup(html, 'html.parser')
    resultados = []

    for fila in soup.find_all('th', scope='row'):
        hora = fila.text.strip()

        # Encuentra el segundo td después del th
        td_super = fila.find_next_sibling('td')         # Primer td (Super Gana)
        td_triple = td_super.find_next_sibling('td')    # Segundo td (Triple Gana)

        # Extrae número y signo de Triple Gana
        numero_tag = td_triple.find('h3', class_='ger')
        numero = numero_tag.text.strip() if numero_tag else None

        signo_img = td_triple.find('img')
        signo = signo_img['src'].split('/')[-1].replace('.jpg', '') if signo_img else None

        # Filtro estricto: solo guardar si hay número y signo válido
        if numero and signo and signo.upper() != 'ppp':
            resultados.append({
                'hora': hora,
                'numero': numero,
                'signo': signo
            })

    return resultados

# Recorre fechas y guarda resultados en la base de datos
def extraer_datos():
    crear_tabla()

    fecha_inicio = datetime(2025, 1, 1)
    fecha_fin = datetime(2025, 10, 31)
    # fecha_fin = datetime.today() - timedelta(days=1)
    fecha_actual = fecha_inicio

    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime("%d/%m/%Y")
        print(f"Consultando {fecha_str} para Triple Gana")

        html = obtener_html(fecha_str)
        resultados = extraer_resultados(html)

        for r in resultados:
            guardar_resultado(fecha_str, r['hora'], r['numero'], r['signo'])

        print("Esperando 30 segundos para la próxima consulta...")
        time.sleep(30)
        fecha_actual += timedelta(days=1)

# Ejecutar directamente
if __name__ == "__main__":
    extraer_datos()
