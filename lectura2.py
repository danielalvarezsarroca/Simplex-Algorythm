import numpy as np
import re

def leer_lineas(archivo):
    with open(archivo, "r") as archivo_txt:
        return archivo_txt.readlines()

def encontrar_seccion(lineas, ID_ALUMNO, ID_PROBLEMA):
    alumno_encontrado = False

    for num_linea, contenido in enumerate(lineas):
        if not alumno_encontrado and re.search(ID_ALUMNO, contenido):
            alumno_encontrado = True
        elif alumno_encontrado and re.search(ID_PROBLEMA, contenido):
            return num_linea  # Retorna el índice donde empieza el problema
    return None  # Si no encuentra el problema, devuelve None

def extraer_vector(lineas, indice):
    if "Columns" in lineas[indice + 2]:  # Caso especial con división de columnas
        valores = ([int(x) for x in lineas[indice + 4].split()] + 
                   [int(x) for x in lineas[indice + 8].split()])
    else:
        valores = [int(x) for x in lineas[indice + 1].split()]
    return np.array(valores)

def extraer_matriz(lineas, indice):
    matriz = []
    linea_actual = 1  # Se empieza en la línea siguiente a 'A='

    if "Columns" in lineas[indice + 2]:  # Caso especial con división en columnas
        matriz_parcial = []
        linea_actual += 3  # Saltar la línea de "Columns X to Y"

        while len(lineas[indice + linea_actual].strip()) > 3:
            matriz_parcial.append([int(valor) for valor in lineas[indice + linea_actual].split()])
            linea_actual += 1

        linea_actual += 3  # Saltar la siguiente línea "Columns X to Y"

        for fila in matriz_parcial:
            matriz.append(fila + [int(valor) for valor in lineas[indice + linea_actual].split()])
            linea_actual += 1
    else:  # Caso normal sin división en columnas
        while len(lineas[indice + linea_actual].strip()) > 3:
            matriz.append([int(valor) for valor in lineas[indice + linea_actual].split()])
            linea_actual += 1

    return np.array(matriz)

def obtener_datos_optimizacion(alumno_id, problema_num, archivo):
    lineas = leer_lineas(archivo)
    
    patron_alumno = rf"alumno\s*{alumno_id}"
    patron_problema = rf"problema\s*PL\s*{problema_num}"

    inicio_problema = encontrar_seccion(lineas, patron_alumno, patron_problema)
    if inicio_problema is None:
        raise ValueError(f"No se encontró el problema {problema_num} para el alumno {alumno_id}")

    coeficientes_c, matriz_A, vector_b = None, None, None

    for linea_idx in range(inicio_problema, len(lineas)):
        contenido = lineas[linea_idx]

        if "c=" in contenido:
            coeficientes_c = extraer_vector(lineas, linea_idx)

        elif "A=" in contenido:
            matriz_A = extraer_matriz(lineas, linea_idx)

        elif "b=" in contenido:
            vector_b = np.array([int(valor) for valor in lineas[linea_idx + 1].split()])
            break  # Se detiene después de leer 'b='

    return coeficientes_c, matriz_A, vector_b

