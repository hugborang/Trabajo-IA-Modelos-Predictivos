import pandas as pd
import numpy as np
import funciones.RobustEvaluation as re


def backward_sequential_search(data, objective, model, N_EXP=5, cV=10):
  
    """
    Realiza búsqueda secuencial hacia atrás para encontrar el mejor subconjunto de variables.

    :param data: pd.DataFrame, Conjunto de datos con N variables predictoras y una variable respuesta.
    :param objective: str, Nombre de la variable respuesta.
    :param model: RandomForestClassifier
    :param N_Exp: int, Número de repeticiones del experimento por validación cruzada (default 5).
    :param CV: int, Número de pliegues (folds) a considerar en la validación cruzada (default 10).
    :return: pd.DataFrame, Tabla con las combinaciones obtenidas en cada iteración, su tamaño y su rendimiento.
    """
    variables = list(data.columns)
    variables.remove(objective)
    current_solution = variables.copy()
    y = data[objective]
    results = []

    for k in range(len(current_solution), 0, -1):
        best_score = -np.inf
        worst_variable = None

        for v in current_solution:
            temp_solution = current_solution.copy()
            temp_solution.remove(v)
            X_temp = data[temp_solution]

            if X_temp.empty:  # Asegurarse de que el DataFrame no esté vacío
                continue
            
            score = re.robust_evaluation(X_temp, y, model,  N_EXP, cV)

            if score > best_score:
                best_score = score
                worst_variable = v
        
        if worst_variable is None:  # Si no se encuentra una variable para eliminar, terminar el bucle
            break

        #Mejor solución temporal
        current_solution.remove(worst_variable)
        results.insert(0, {
            'variables': current_solution.copy(),
            'size': len(current_solution),
            'score': best_score
        })

    all_variablesScores = re.robust_evaluation(data[variables], y, model, N_EXP, cV)

    results.insert(-1, {
        'variables': variables,
        'size': len(variables),
        'score': all_variablesScores
    })
    
    sorted_results = pd.DataFrame(results)
    sorted_results = sorted_results.sort_values(by='score', ascending=False)
    
    return sorted_results    


