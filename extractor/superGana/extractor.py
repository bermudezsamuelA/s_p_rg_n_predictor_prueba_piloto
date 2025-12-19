# extractor.py — Extrae resultados de Super Gana con robustez y logging mejorado

import requests
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import sys
import os
import pandas as pd

# Configurar logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('supergana_extractor.log'),
        logging.StreamHandler()
    ]
)

# Agregar el directorio actual al path para importar db.py
sys.path.append(os.path.dirname(__file__))
from db import guardar_resultado, crear_tabla

def obtener_html(fecha_str, max_reintentos=3):
    """Obtiene HTML de una fecha específica con reintentos inteligentes."""
    url = f"https://supergana.com.ve/pruebah.php?bt={fecha_str.replace('/', '%2F')}"
    
    for intento in range(max_reintentos):
        try:
            respuesta = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; SuperGanaBot/1.0)'}
            )
            respuesta.raise_for_status()
            
            if "<th" not in respuesta.text:  # Validación mínima
                raise ValueError("Respuesta HTML inválida o vacía")
            
            return respuesta.text
        
        except Exception as e:
            logging.warning(f"{fecha_str}: Error en intento {intento+1}/{max_reintentos} → {e}")
            if intento < max_reintentos - 1:
                espera = 2 ** intento
                time.sleep(espera)
    
    logging.error(f"{fecha_str}: Fallo tras {max_reintentos} intentos")
    return None

def extraer_resultados(html, fecha_str):
    """Parsea el HTML y devuelve resultados válidos."""
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    resultados = []
    horas_validas = {'1 pm', '4 pm', '10 pm'}
    
    for fila in soup.find_all('th', scope='row'):
        hora = fila.text.strip().lower()
        if hora not in horas_validas:
            continue
        
        td = fila.find_next_sibling('td')
        if not td:
            continue
        
        numero_tag = td.find('h3', class_='ger')
        if not numero_tag:
            continue
        
        numero = numero_tag.text.strip()
        if not numero.isdigit() or len(numero) != 4:
            continue
        
        signo_img = td.find('img')
        if not signo_img or 'src' not in signo_img.attrs:
            continue
        
        signo = signo_img['src'].split('/')[-1].replace('.jpg', '').upper()
        if signo == 'PPP' or len(signo) != 3:
            continue
        
        resultados.append({'hora': hora, 'numero': numero, 'signo': signo})
    
    return resultados

def extraer_datos(fecha_inicio=None, fecha_fin=None, pausa=2):
    """Extrae datos en un rango de fechas y guarda en SQLite."""
    crear_tabla()
    
    if fecha_inicio is None:
        fecha_inicio = datetime(2025, 1, 1)
    if fecha_fin is None:
        fecha_fin = datetime.today() - timedelta(days=1)
    
    fecha_actual = fecha_inicio
    total_resultados, total_guardados = 0, 0
    fechas_con_error = []
    
    logging.info(f"Extracción Super Gana desde {fecha_inicio:%d/%m/%Y} hasta {fecha_fin:%d/%m/%Y}")
    
    while fecha_actual <= fecha_fin:
        fecha_str = fecha_actual.strftime("%d/%m/%Y")
        html = obtener_html(fecha_str)
        
        if html is None:
            fechas_con_error.append(fecha_str)
            fecha_actual += timedelta(days=1)
            continue
        
        resultados = extraer_resultados(html, fecha_str)
        total_resultados += len(resultados)
        
        guardados_dia = 0
        for r in resultados:
            if guardar_resultado(fecha_str, r['hora'], r['numero'], r['signo']):
                guardados_dia += 1
                logging.debug(f"Guardado {fecha_str} {r['hora']} {r['numero']} {r['signo']}")
            else:
                logging.debug(f"Duplicado {fecha_str} {r['hora']} {r['numero']} {r['signo']}")
        
        if guardados_dia > 0:
            total_guardados += guardados_dia
            logging.info(f"{fecha_str}: {guardados_dia} resultado(s) guardado(s)")
        elif resultados:
            logging.info(f"{fecha_str}: {len(resultados)} resultado(s) encontrados (ya existían)")
        else:
            logging.info(f"{fecha_str}: Sin resultados")
        
        time.sleep(pausa)
        fecha_actual += timedelta(days=1)
    
    logging.info("=" * 50)
    logging.info(f"Total encontrados: {total_resultados}")
    logging.info(f"Total guardados: {total_guardados}")
    logging.info(f"Fechas con error: {len(fechas_con_error)}")
    
    if fechas_con_error:
        pd.DataFrame(fechas_con_error, columns=['fecha']).to_csv('errores_supergana.csv', index=False)
        logging.warning(f"Errores exportados a errores_supergana.csv")
    
    return {
        'total_resultados': total_resultados,
        'total_guardados': total_guardados,
        'fechas_con_error': fechas_con_error
    }

def main():
    """CLI para ejecutar el extractor"""
    import argparse
    parser = argparse.ArgumentParser(description='Extractor de resultados de Super Gana')
    parser.add_argument('--inicio', help='Fecha inicial (DD/MM/YYYY)')
    parser.add_argument('--fin', help='Fecha final (DD/MM/YYYY)')
    parser.add_argument('--hoy', action='store_true', help='Extraer solo ayer')
    parser.add_argument('--pausa', type=int, default=2, help='Segundos de espera entre consultas')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.hoy:
        fecha_fin = datetime.today() - timedelta(days=1)
        fecha_inicio = fecha_fin
    else:
        fecha_inicio = datetime.strptime(args.inicio, '%d/%m/%Y') if args.inicio else None
        fecha_fin = datetime.strptime(args.fin, '%d/%m/%Y') if args.fin else None
    
    extraer_datos(fecha_inicio, fecha_fin, pausa=args.pausa)

if __name__ == "__main__":
    main()
