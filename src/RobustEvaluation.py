import numpy as np

from sklearn.model_selection import cross_val_score

def robust_evaluation(model, X, y, N_Exp=1, CV=3, Scoring='balanced_accuracy'):

    #print("Debugging: Columns of X:", X.columns)
    #print("DATAFRAME:", X)
    #print("solution:", solution )
    """
    Evalúa un modelo utilizando validación cruzada repetida y devuelve la métrica de rendimiento promedio.

    :param model: sklearn model, Instancia del modelo de entrenamiento a usar.
    :param X: pd.DataFrame, Conjunto de datos con variables predictoras.
    :param y: pd.Series, Variable respuesta.
    :param N_Exp: int, Número de repeticiones del experimento.
    :param CV: int, Número de pliegues (folds) para la validación cruzada.
    :param scoring: str or callable, Estrategia para evaluar el rendimiento del modelo.
    :return: float, Métrica de rendimiento promedio obtenida.
    """
    scores = []

    for _ in range(N_Exp):
        cv_results = cross_val_score(model, X, y, cv=CV, scoring=Scoring, n_jobs=-1)
        scores.extend(cv_results)


    return np.mean(scores)
