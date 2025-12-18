# s_p_rg_n_predictor_prueba_piloto
la intencion es generar un sistema de estudio, y/o prediccion

cosas a hacer

Aquí está mi plan de ataque:

    Limpiar y unir los datos de Super y Triple Gana

    Explorar frecuencias, paridad, rangos, signos, combinaciones

    Aplicar chi-cuadrado para detectar dependencias

    Usar series de tiempo para ver ciclos o repeticiones

    Visualizar con heatmaps, boxplots y scatterplots

    Exportar los hallazgos y preparar para modelado predictivo

     Estrategia general de análisis
1. Exploración descriptiva

Antes de aplicar pruebas estadísticas, primero entender la distribución:

    Frecuencia de aparición de cada número → ¿cuáles se repiten más?

    Frecuencia por hora (1 pm, 4 pm, 10 pm) → ¿hay números que dominan en ciertos horarios?

    Distribución de signos zodiacales → ¿hay signos que aparecen más con ciertos números?

    Paridad (pares vs impares) → ¿hay un sesgo hacia pares?

    Rangos (bajos vs altos) → ¿más números entre 0000–4999 o 5000–9999?

2. Pruebas estadísticas para sesgos
📊 Chi-cuadrado de independencia

Ideal para responder:

    ¿La hora del sorteo influye en el número que aparece?

    ¿El signo zodiacal está asociado con la paridad del número?

    ¿Hay dependencia entre día del mes y tipo de número?
3. Series de tiempo

Ideal para ver si hay ciclos o repeticiones:

    ¿Ciertos números aparecen cada X días?

    ¿Hay signos que se repiten en patrones semanales?

    ¿La frecuencia cambia con el mes?

Usar pandas con resample() y rolling() para suavizar y detectar tendencias.
4. Combinaciones y correlaciones

    ¿Número + signo tienen combinaciones frecuentes?

    ¿Hay signos que aparecen con ciertos rangos de número?

    ¿Algún número aparece con más de un signo?

Usar groupby(['numero', 'signo']) y contar ocurrencias.
5. Visualización avanzada

Algunas gráficas son difíciles de leer porque hay muchos valores. Para mejorar:

    Usar heatmaps para combinaciones (por ejemplo, número vs hora).

    Usar boxplots para ver distribución por grupo (por ejemplo, número por signo).

    Usar scatterplots con color por signo para ver agrupaciones.

🔗 ¿Y qué hacemos con Triple Gana?

Tu intuición es excelente. Si ambas loterías:

    Se publican a las mismas horas

    Tienen el mismo formato (número + signo)

    Son del mismo operador

Entonces sí vale la pena analizarlas en paralelo. Aquí hay tres formas de hacerlo:
A. Comparación directa

    ¿Triple Gana repite números de Super Gana?

    ¿Comparten signos en el mismo horario?

B. Correlación cruzada

    ¿El número de Triple Gana predice el de Super Gana?

    ¿Hay sincronía entre ambas series?

C. Análisis conjunto

    Unir ambas tablas y agregar una columna juego = 'super' | 'triple'

    Ver si hay sesgos distintos por juego


📁 data/
│   ├── supergana.db
│   └── triplegana.db

📁 extractor/
│   ├── 📁 supergana/
│   │   ├── db.py
│   │   └── extractor.py
│   └── 📁 triplegana/
│       ├── db.py
│       └── extractor.py

📁 predictor/
│   ├── analyzer_comparativo.py
│   ├── estadisticas.py
│   ├── exportador.py
│   ├── series_tiempo.py
│   ├── visualizaciones.py
│   ├── 📁 supergana/
│   │   └── analyzer.py
│   └── 📁 triplegana/
│       └── analyzer.py

