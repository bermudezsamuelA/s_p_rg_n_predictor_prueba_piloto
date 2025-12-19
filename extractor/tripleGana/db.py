import sqlite3
import os

# Obtener ruta absoluta para evitar problemas de directorio
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'triplegana.db')

def crear_tabla():
    """Crea la tabla específica para Triple Gana"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS triplegana_resultados (
            fecha TEXT NOT NULL,       -- Fecha del sorteo (ej. "09/10/2025")
            hora TEXT NOT NULL,        -- Hora del sorteo (ej. "1 pm")
            numero TEXT NOT NULL,      -- Número sorteado (ej. "8144")
            signo TEXT NOT NULL,       -- Signo zodiacal (ej. "CAP")
            extraido_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (fecha, hora)  -- Evita duplicados por fecha/hora
        )
    ''')
    
    # Índice adicional para acelerar consultas por fecha y hora
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_triplegana_fecha_hora ON triplegana_resultados(fecha, hora)')
    
    conn.commit()
    conn.close()

def guardar_resultado(fecha, hora, numero, signo):
    """Guarda un resultado evitando duplicados"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO triplegana_resultados 
            (fecha, hora, numero, signo) 
            VALUES (?, ?, ?, ?)
        ''', (fecha, hora, numero, signo))
        
        conn.commit()
        # lastrowid es más confiable que rowcount en SQLite
        return cursor.lastrowid is not None
    except sqlite3.Error as e:
        print(f"Error SQLite: {e}")
        return False
    finally:
        conn.close()
