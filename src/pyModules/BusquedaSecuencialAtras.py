
import pandas as pd
import numpy as np
import RobustEvaluation as re


def backward_sequential_search(data, target_name, model, N_Exp=10, CV=5, scoring='neg_mean_squared_error'):
    """
    Realiza búsqueda secuencial hacia atrás para encontrar el mejor subconjunto de variables.

    :param data: pd.DataFrame, Conjunto de datos con N variables predictoras y una variable respuesta.
    :param target_name: str, Nombre de la variable respuesta.
    :param model: sklearn model, Instancia del modelo de entrenamiento a usar.
    :param N_Exp: int, Número de repeticiones del experimento por validación cruzada (default 10).
    :param CV: int, Número de pliegues (folds) a considerar en la validación cruzada (default 5).
    :param scoring: str or callable, Estrategia para evaluar el rendimiento del modelo (default 'neg_mean_squared_error').
    :return: pd.DataFrame, Tabla con las combinaciones obtenidas en cada iteración, su tamaño y su rendimiento.
    """
    # Inicialización
    variables = list(data.columns)
    variables.remove(target_name)
    current_solution = variables.copy()
    y = data[target_name]
    results = []

    # Ejecución
    for k in range(len(current_solution), 0, -1):
        best_score = -np.inf
        worst_variable = None

        # Evaluar cada variable
        for variable in current_solution:
            temp_solution = current_solution.copy()
            temp_solution.remove(variable)
            score = re.robust_evaluation(model, data[temp_solution], y, N_Exp, CV, scoring)

            if score > best_score:
                best_score = score
                worst_variable = variable

        # Actualizar la solución actual
        current_solution.remove(worst_variable)
        results.append({
            'variables': current_solution.copy(),
            'size': len(current_solution),
            'score': best_score
        })

    # Devolver los resultados en formato de DataFrame
    return pd.DataFrame(results)

