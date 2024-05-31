import pandas as pd
import numpy as np
import RobustEvaluation as re

def backward_sequential_mixed_search_(data, objective, model, N_Exp, cV, M):
   
   
    x = list(data.columns)
    x.remove(objective)
    y = data[objective]
    
    # Inicialización
    current_solution = x
    Añadidos = []
    Eliminados = []
    results = []
    contador = 0


    #Falta condicion de parada
    while  len(Eliminados) != len(x) and contador < M:


        if(len(Eliminados) == len(x)):
            contador +=1

        best_score1 = -np.inf
        worst_variable = None

        for variable in x:
            if variable not in Eliminados:
                temp_solution = current_solution.copy()
                temp_solution.remove(variable)
                X_temp = data[temp_solution]
                    
                if X_temp.empty:  
                    continue

                score = re.robust_evaluation(model, X_temp, y, N_Exp, cV)

                if score > best_score1:
                    best_temp_solution = current_solution
                    worst_variable = variable
                    best_score1 = score


        if worst_variable:
            Añadidos.append(worst_variable)
            Eliminados.append(worst_variable)
            contador = 0


        best_score2 = best_score1
        best_variable = None


        for variable in x:
            if variable not in temp_solution and variable not in Añadidos:
                temp_solution = current_solution.copy()
                temp_solution.append(variable)
                X_temp = data[temp_solution]
                
                if X_temp.empty:  
                    continue

                score = re.robust_evaluation(model, X_temp, y, N_Exp, cV)

                #Como best_score2 = best_score1, se puede omitir la condicion score > best_score1
                if score > best_score2:
                    best_temp_solution = temp_solution
                    best_variable = variable
                    best_score2 = score

        if best_variable:
            contador = 0
            Añadidos.append(best_variable)

        results.insert(0, {
                'variables': best_temp_solution.copy(),
                'size': len(best_temp_solution),
                'score': best_score2
            })
        
        
    return pd.DataFrame(results)
    

