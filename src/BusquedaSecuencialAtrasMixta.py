import pandas as pd
import numpy as np
import RobustEvaluation as re


def backward_sequential_mixed_search_(data, target_name, model, N_Exp, cV, M=10):
    """
    Realiza búsqueda secuencial hacia atrás para encontrar el mejor subconjunto de variables.

    :param data: pd.DataFrame, Conjunto de datos con N variables predictoras y una variable respuesta.
    :param target_name: str, Nombre de la variable respuesta.
    :param model: sklearn model, Instancia del modelo de entrenamiento a usar.
    :param N_Exp: int, Número de repeticiones del experimento por validación cruzada (default 1).
    :param M: int, Umbral de iteraciones sin mejoras para la condición de parada (default 10).
    :return: pd.DataFrame, Tabla con las combinaciones obtenidas en cada iteración, su tamaño y su rendimiento.
    """
    # Inicialización
    variables = list(data.columns)
    variables.remove(target_name)
    current_solution = variables.copy()
    y = data[target_name]
    results = []
    counter = 0

    # Ejecución
    while counter < M:
        best_score = -np.inf
        worst_variable = None

        # Eliminar la peor variable
        for variable in current_solution:
            temp_solution = current_solution.copy()
            temp_solution.remove(variable)
            X_temp = data[temp_solution]

            score = re.cross_val_score(model, X_temp, y, N_Exp, cV)

            if score > best_score:
                best_score = score
                worst_variable = variable

        if worst_variable is not None:
            current_solution.remove(worst_variable)
            results.insert(0, {
                'variables': current_solution.copy(),
                'size': len(current_solution),
                'score': best_score
            })
            counter = 0
        else:
            counter += 1

    return pd.DataFrame(results)
