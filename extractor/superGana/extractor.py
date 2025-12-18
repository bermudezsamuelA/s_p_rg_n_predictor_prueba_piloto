# extractor.py — Encargado de consultar el endpoint, extraer resultados y guardarlos en SQLite

import requests                      # Para hacer llamadas HTTP al endpoint
import time                          # Para pausar entre llamadas y evitar sobrecargar el servidor
from datetime import datetime, timedelta  # Para recorrer fechas fácilmente
from bs4 import BeautifulSoup        # Para parsear el HTML y extraer datos
from db import guardar_resultado, crear_tabla  # Funciones que gestionan la base de datos

# Función que construye la URL y obtiene el HTML de una fecha específica
def obtener_html(fecha_str):
    url = f"https://supergana.com.ve/pruebah.php?bt={fecha_str.replace('/', '%2F')}"
    response = requests.get(url)
    return response.text

# Función que parsea el HTML y extrae los resultados válidos
def extraer_resultados(html):
    soup = BeautifulSoup(html, 'html.parser')
    resultados = []

    # Recorre cada fila que representa una hora del sorteo
    for fila in soup.find_all('th', scope='row'):
        hora = fila.text.strip()  # Extrae la hora (ej. "1 pm")
        td = fila.find_next_sibling('td')  # Encuentra la celda con los datos

        # Extrae el número si existe
        numero_tag = td.find('h3', class_='ger')
        numero = numero_tag.text.strip() if numero_tag else None

        # Extrae el signo zodiacal desde la imagen
        signo_img = td.find('img')
        signo = signo_img['src'].split('/')[-1].replace('.jpg', '') if signo_img else None

        # Filtro estricto: solo guardar si hay número y el signo no es 'ppp'
        if numero and signo and signo.upper() != 'ppp':
            resultados.append({
                'hora': hora,
                'numero': numero,
                'signo': signo
            })

    return resultados

# Función principal que recorre fechas, consulta el endpoint y guarda los resultados
def extraer_datos():
    crear_tabla()  # Asegura que la tabla exista antes de insertar

    # Define el rango de fechas a consultar
    fecha_inicio = datetime(2025, 1, 1)
    # fecha_fin = datetime(2025, 10, 30)
    fecha_fin = datetime.today() - timedelta(days=1)
    fecha_actual = fecha_inicio

    # Recorre cada día dentro del rango
    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime("%d/%m/%Y")
        print(f"Consultando {fecha_str}")

        html = obtener_html(fecha_str)               # Obtiene el HTML del día
        resultados = extraer_resultados(html)        # Extrae los resultados válidos

        # Guarda cada resultado en la base de datos
        for r in resultados:
            guardar_resultado(fecha_str, r['hora'], r['numero'], r['signo'])
        print("Esperando 30 segundos para la próxima consulta...")
        time.sleep(30)  # Pausa para no saturar el servidor
        fecha_actual += timedelta(days=1)  # Avanza al siguiente día

# Permite ejecutar el script directamente
if __name__ == "__main__":
    extraer_datos()
