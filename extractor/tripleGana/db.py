# db.py — Encargado de crear la tabla y guardar los resultados en SQLite

import sqlite3  # Librería nativa de Python para trabajar con bases de datos SQLite

# Crea la tabla 'resultados' si no existe
def crear_tabla():
    conn = sqlite3.connect('data/triplegana.db')  # Conecta (o crea) el archivo de base de datos
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resultados (
            fecha TEXT,     -- Fecha del sorteo (ej. "09/10/2025")
            hora TEXT,      -- Hora del sorteo (ej. "1 pm")
            numero TEXT,    -- Número sorteado (ej. "8144")
            signo TEXT      -- Signo zodiacal (ej. "CAP")
        )
    ''')

    conn.commit()  # Guarda los cambios
    conn.close()   # Cierra la conexión

# Inserta un resultado en la tabla
def guardar_resultado(fecha, hora, numero, signo):
    conn = sqlite3.connect('data/triplegana.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO resultados (fecha, hora, numero, signo)
        VALUES (?, ?, ?, ?)
    ''', (fecha, hora, numero, signo))  # Inserta los valores usando parámetros seguros

    conn.commit()
    conn.close()
