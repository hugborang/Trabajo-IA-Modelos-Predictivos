import numpy as np

from sklearn.model_selection import cross_val_score

def robust_evaluation(X, y, model, N_Exp, CV):

    """
    Evalúa un modelo utilizando validación cruzada repetida y devuelve la métrica de rendimiento promedio.

    :param X: pd.DataFrame, Conjunto de datos con variables predictoras.
    :param y: pd.Series, Variable respuesta.
    :param model: RandomForestClassifier
    :param N_Exp: int, Número de repeticiones del experimento.
    :param CV: int, Número de pliegues (folds) para la validación cruzada.
    :return: float, Métrica de rendimiento promedio obtenida.
    """
    scores = []

    #Usamos balanced_accuracy como métrica de rendimiento y usamos todos los núcleos disponibles
    for _ in range(N_Exp):
        cv_results = cross_val_score(model, X, y, cv=CV, scoring='balanced_accuracy', n_jobs=-1)
        scores.extend(cv_results)

    return np.mean(scores)
