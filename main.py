from lectura2 import *
import numpy as np

# El código está estructurado en tres partes:
    # 1. Funciones auxiliares para la ejecución del algoritmo; cálculo z, theta, etc...
    # 2. Funciones del algoritmo Simplex; inicialización fase 1 y 2 + BUCLE PRINCIPAL
    # 3. Ejecución del código, llamando a la lectura de datos y a las funciones del algoritmo Simplex


###################################################################################################################
# PARTE 1: FUNCIONES AUXILIARES PARA EL ALGORITMO PRINCIPAL
####################################################################################################################

def calculo_z(C_b, X_b, C_n, X_n):

    #################
    # Cálculo de la z
    #################

    z_basicas = (C_b @ X_b).sum()
    z_no_basicas = (C_n @ X_n).sum()
    z = z_basicas + z_no_basicas
    return z


def calcular_costes_reducidos(C_n, C_b, B_inv, A_n):
    #################################
    # Cálculo de los costes reducidos
    #################################
    x = C_b @ B_inv
    y = x @ A_n
    costes_reducidos = C_n - y
    return costes_reducidos


def variable_entrada(costes_reducidos, vars_no_basicas):
    #####################################
    # Selección de la variable de entrada
    #####################################
    i = 0
    indices_negativos = []
    
    while i < len(costes_reducidos):
        if costes_reducidos[i] < 0:
            indices_negativos.append(i)
        i += 1

    posibles = [vars_no_basicas[k] for k in indices_negativos]

    if len(posibles) > 0:
        return min(posibles)  # Regla de Bland
    return None



def direccion_basica_factible(B_inv, A_n, var_entrada, vars_no_basicas):
    #########################################
    # Cálculo de la dirección básica factible
    #########################################

    indice_var_entrada = vars_no_basicas.index(var_entrada) 
    D_b = -np.dot(B_inv, A_n[:, indice_var_entrada])
    return D_b


def theta(X_b, D_b, vars_basicas):
    ##########################################################
    # Cálculo de la theta y selección de la variable de salida
    ##########################################################
    i = 0
    candidatos = []

    while i < len(D_b):
        if D_b[i] < 0:
            valor_theta = -X_b[i] / D_b[i]
            if valor_theta > 0:
                candidatos.append((i, valor_theta))
        i += 1

    if candidatos:
        minimo = min(valor for _, valor in candidatos)
        posibles_indices = [indice for indice, valor in candidatos if valor == minimo]
        mejor_indice = min(posibles_indices, key=lambda idx: vars_basicas[idx])
        return vars_basicas[mejor_indice], minimo
    else:
        return None, np.inf

def actualizacion(vars_basicas, vars_no_basicas, X_b, theta, direccion, z, costes_reducidos, C_b, C_n, c, A_n, A_ampliada, B_inv, var_entrada, var_salida):
    #######################################################################
    # Función para actualizar todos los parámetros después de una iteración 
    ########################################################################
    
    pos_salida = vars_basicas.index(var_salida)
    pos_entrada = vars_no_basicas.index(var_entrada)

    # Actualizar variables básicas y no básicas
    vars_basicas[pos_salida] = var_entrada
    vars_no_basicas[pos_entrada] = var_salida

    # Actualizar X_b 
    nueva_X_b = np.copy(X_b)
    for i in range(len(direccion)):
        incremento = theta * direccion[i]
        nueva_X_b[i] = theta if i == pos_salida else (X_b[i] + incremento)

    X_b[:, 0] = nueva_X_b[:, 0]  

    # Actualizar inversa de B
    tamaño = direccion.shape[0]
    E = np.eye(tamaño)
    divisor = direccion[pos_salida]
    for fila in range(tamaño):
        if fila != pos_salida:
            E[fila, pos_salida] = -direccion[fila] / divisor
    E[pos_salida, pos_salida] = 1 / divisor * -1

    B_inv = E @ B_inv

    # Actualizar A_n
    columna_reemplazo = A_ampliada[:, var_salida - 1]
    A_n[:, pos_entrada] = columna_reemplazo

    # Actualizar costes 
    C_b[pos_salida] = c[var_entrada - 1]
    C_n[pos_entrada] = c[var_salida - 1]

    # Actualizar z
    incremento_z = theta * costes_reducidos[pos_entrada]
    z_nuevo = float(z) + float(incremento_z)
    assert z_nuevo < z, "El valor de z no ha disminuido como se esperaba."

    return vars_basicas, vars_no_basicas, X_b, z_nuevo, C_b, C_n, A_n, B_inv


####################################################################################################################
# PARTE 2: INICIALIZACIÓN FASE 1 Y 2 DEL SIMPLEX + BUCLE PRINCIPAL DEL ALGORITMO
#####################################################################################################################

def init_fase_1(A, A_artificial, b):  

    #########################################
    # Inicialización de la Fase 1 del Simplex
    #########################################

    m, n_original = A.shape
    n_total = A_artificial.shape[1]

    artificiales = list(range(n_original + 1, n_total + 1))
    basicas = artificiales.copy()

    no_basicas = list(range(1, n_original + 1))

    c_auxiliar = np.concatenate([np.zeros(n_original), np.ones(n_total - n_original)])

    B = np.eye(m)
    inversa_B = B.copy()  
    A_no_basicas = A.copy()

    costes_basicas = c_auxiliar[n_original:].copy()
    costes_no_basicas = c_auxiliar[:n_original].copy()

    x_basicas = inversa_B @ b
    x_no_basicas = np.zeros((n_original, 1))

    valor_z = calculo_z(costes_basicas, x_basicas, costes_no_basicas, x_no_basicas)

    return basicas, artificiales, no_basicas, c_auxiliar, B, inversa_B, A_no_basicas, costes_basicas, costes_no_basicas, x_basicas, x_no_basicas, valor_z



def init_fase2(vars_basicas, variables_basicas, vars_no_basicas, c, b, inversa_B):

    #########################################
    # Inicialización de la Fase 2 del Simplex
    #########################################

    i = 0
    while i < len(vars_no_basicas):
        if vars_no_basicas[i] in variables_basicas:
            vars_no_basicas.pop(i)
        else:
            i += 1
  
    columnas = np.array(vars_no_basicas) - 1
    A_n = np.take(A, columnas, axis=1)

    X_b = inversa_B @ b
    X_n = np.zeros((len(vars_no_basicas), 1))

    C_b = np.take(c, np.array(vars_basicas) - 1)
    C_n = np.take(c, columnas)

    z = calculo_z(C_b, X_b, C_n, X_n)

    return C_b, C_n, X_b, X_n, A_n, z


def resolver_simplex(fase, matriz_A=None, vector_b=None, iter_inicial=None, basicas=None,
                     originales=None, no_basicas=None, z=None, objetivo=None,
                     B_inv=None, A_ext=None):
    
    ########################################################################################
    # Función que llama a todas las otras creadas para resolver el problema mediante Simplex
    ########################################################################################

    if fase == 1:
        print("-" * 40 + " INICIANDO FASE 1 " + "-" * 40)
        m = A.shape[0]  
        identidad = np.eye(m)  
        A_ext = np.hstack((A, identidad))  
        resultado = init_fase_1(matriz_A, A_ext, vector_b)
        basicas, originales, no_basicas, c_aux, _, B_inv, A_n, Cb, Cn, Xb, _, z = resultado
        iteracion = 1
        print(f"Z inicial: {z:.3f}")
        print(f"Variables básicas iniciales: {basicas}")
        costes = calcular_costes_reducidos(Cn, Cb, B_inv, A_n)
        coste_total = c_aux
    else:
        print("-" * 40 + " INICIANDO FASE 2 " + "-" * 40)
        Cb, Cn, Xb, _, A_n, z = init_fase2(basicas, originales, no_basicas, objetivo, vector_b, B_inv)
        B_inv = B_inv.copy()
        iteracion = iter_inicial
        costes = calcular_costes_reducidos(Cn, Cb, B_inv, A_n)
        coste_total = objetivo

    # Bucle principal del método
    while True:
        if np.all(costes >= 0): # Evalúa si  es óptimo, si todos los costes son mayores que cero
            if fase == 1 and z > 1e-10:
                print("No se ha encontrado una solución factible.")
                return None
            break

        col_entrada = variable_entrada(costes, no_basicas)
        if col_entrada is None:
            print("Óptimo alcanzado.")
            break

        direccion = direccion_basica_factible(B_inv, A_n, col_entrada, no_basicas)
        col_salida, paso = theta(Xb, direccion, basicas)

        if col_salida is None or paso == np.inf:
            print("Problema no acotado.")
            return None

        args_actualizacion = (
            basicas, no_basicas, Xb, paso, direccion, z, costes,
            Cb, Cn, coste_total, A_n, A_ext, B_inv, col_entrada, col_salida
        )
        resultado_act = actualizacion(*args_actualizacion)
        basicas, no_basicas, Xb, z, Cb, Cn, A_n, B_inv = resultado_act

        paso_valor = float(paso.item()) if hasattr(paso, "item") else float(paso)
        print(f"Iteración {iteracion}: v_entrada = {col_entrada}, v_salida = {col_salida}, θ = {paso_valor:.3f}, z = {z:.3f}")
        iteracion += 1

        costes = calcular_costes_reducidos(Cn, Cb, B_inv, A_n)

    if fase == 1:
        print(f"Fase 1 completada tras {iteracion - 1} iteraciones.")
        return iteracion, basicas, originales, no_basicas, A_n, Cb, Cn, Xb, z, B_inv, A_ext

    print(f"Fase 2 completada tras {iteracion - 1} iteraciones.")
    print("-" * 100)
    print("-" * 40 + " RESULTADOS " + "-" * 40)
    print(f"z*: {z:.6f}")
    print(f"Vb*: {basicas}")
    print(f"Valores básicos: {Xb.flatten()}")
    print(f"r*: {costes}")
    return basicas, no_basicas, B_inv, A_n, Cb, Cn, Xb, z


def ejecutar_fase2(resultado_fase1, funcion_objetivo, vector_b):

    #################################################################################
    # Función para ejecutar de la Fase 2 del Simplex, con los resultados de la Fase 1
    #################################################################################

    iteraciones, vb, vb_original, vnb, A_n, Cb, Cn, Xb, z, inversa_B, A_ext = resultado_fase1

    return resolver_simplex(
        fase=2,
        iter_inicial=iteraciones,
        basicas=vb,
        originales=vb_original,
        no_basicas=vnb,
        z=z,
        objetivo=funcion_objetivo,
        vector_b=vector_b,
        B_inv=inversa_B,
        A_ext=A_ext
    )


#####################################################################################################################
# PARTE 3: EJECUCIÓN DEL CÓDIGO	
######################################################################################################################

if __name__ == "__main__":

    print("SIMPLEX ALGORYTHM".center(100))
    print("-" * 100)

    try:
        num_alumno = int(input("Introduce el número de alumno: "))
        problema = int(input("Introduce el número de problema: "))
    except ValueError:
        print("Entrada no válida. Asegúrate de introducir números enteros.")
        exit()

    archivo_datos = "problemas.txt"

    datos = obtener_datos_optimizacion(num_alumno, problema, archivo_datos)
    if datos is None:
        print("No se han podido obtener los datos del problema.")
        exit()

    funcion_objetivo, A, b = datos

    if A is None or b is None:
        print("Error en la carga de la matriz o el vector de términos independientes.")
        exit()

    b = b.reshape(-1, 1)

    print("-" * 100)
    print("RESOLVIENDO...".center(100))
    print("-" * 100)

    resultado_fase1 = resolver_simplex(fase=1, matriz_A=A, vector_b=b)

    if resultado_fase1 is not None:
        ejecutar_fase2(resultado_fase1, funcion_objetivo, b)
    else:
        print("Fase 1 ha determinado que el problema no es factible. THE END.")
