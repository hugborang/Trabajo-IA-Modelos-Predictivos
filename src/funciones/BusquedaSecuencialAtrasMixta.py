import pandas as pd
import numpy as np
import funciones.RobustEvaluation as re

def backward_sequential_mixed_search(data, objective, model, N_exp, cV, M):
   
    variables = list(data.columns)
    variables.remove(objective)
    y = data[objective]
    
    # Inicialización
    current_solution = variables.copy()
    adds= []
    deleted = []
    results = []
    counter = 0

    while  len(deleted) != len(variables) or counter < M:
        
        if(len(deleted) == len(variables)):
            counter +=1
       
        best_score = -np.inf
        worst_variable = None

        for v in variables:
            if v not in deleted:
                temp_solution = current_solution.copy()
                temp_solution.remove(v)
                X_temp = data[temp_solution]

                score = re.robust_evaluation(X_temp, y, model, N_exp, cV)

                if score > best_score:
                    worst_variable = v
                    best_score = score

        if worst_variable:
            current_solution.remove(worst_variable)
            deleted.append(worst_variable)

        #Añadir la mejor variable solo si hay mejora
        best_variable = None

        for v in variables:
            if v not in current_solution and v not in adds:
                temp_solution = current_solution.copy()
                temp_solution.append(v)
                X_temp = data[temp_solution]
                
                score = re.robust_evaluation(X_temp, y, model,  N_exp, cV)

                if score > best_score:
                    best_variable = v
                    best_score = score

        if best_variable:
            current_solution.append(best_variable)
            adds.append(best_variable)
            counter = 0

        #Para evitar que se añada una variable que no mejora el modelo.
        if(score != -np.inf):
            results.append({
                    'variables': current_solution.copy(),
                    'size': len(current_solution),
                    'score': best_score
                })
        
    sorted_results = pd.DataFrame(results)
    sorted_results = sorted_results.sort_values(by='score', ascending=False)

    return sorted_results    
