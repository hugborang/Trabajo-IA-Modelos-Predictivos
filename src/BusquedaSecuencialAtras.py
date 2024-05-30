import pandas as pd
import numpy as np
import RobustEvaluation as re


def backward_sequential_search(data, target_name, model, N_EXP, cV):
  
    """
    Realiza búsqueda secuencial hacia atrás para encontrar el mejor subconjunto de variables.

    :param data: pd.DataFrame, Conjunto de datos con N variables predictoras y una variable respuesta.
    :param target_name: str, Nombre de la variable respuesta.
    :param model: sklearn model, Instancia del modelo de entrenamiento a usar.
    :param N_Exp: int, Número de repeticiones del experimento por validación cruzada (default 1).
    :param CV: int, Número de pliegues (folds) a considerar en la validación cruzada (default 3).
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
            X_temp = data[temp_solution]
            
            if X_temp.empty:  # Asegurarse de que el DataFrame no esté vacío
                continue

            score = re.robust_evaluation(model, X_temp, y, N_EXP, cV)

            if score > best_score:
                best_score = score
                worst_variable = variable

        if worst_variable is None:  # Si no se encuentra una variable para eliminar, terminar el bucle
            break

       # Actualizar la solución actual
        current_solution.remove(worst_variable)
        results.insert(0, {
            'variables': current_solution.copy(),
            'size': len(current_solution),
            'score': best_score
        })


    # Devolver los resultados en formato de DataFrame
    return pd.DataFrame(results)
