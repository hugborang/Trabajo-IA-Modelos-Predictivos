import pandas as pd
import numpy as np
import RobustEvaluation as re


def backward_sequential_search(data, target_name, model):
  
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

            score = re.robust_evaluation(model, X_temp, y)

            if score > best_score:
                best_score = score
                worst_variable = variable

        if worst_variable is None:  # Si no se encuentra una variable para eliminar, terminar el bucle
            break

        # Actualizar la solución actual
        current_solution.remove(worst_variable)
        results.append({
            'variables': current_solution.copy(),
            'size': len(current_solution),
            'score': best_score
        })

    # Devolver los resultados en formato de DataFrame
    print("RESULTS:", results)
    return pd.DataFrame(results)




if __name__ == "__main__":

    import pandas as pd
    import BusquedaSecuencialAtras as bsa
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    titanic = pd.read_csv('./data/titanic.csv')
    titanic.head()

    atributos_discretos = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Initial', 'Age_band', 
    'Family_Size', 'Alone', 'Fare_cat', 'Deck', 'Title', 'Is_Married','Survived']
    atributos_continuos = ['Age', 'Fare']
    atributos = titanic.loc[:, atributos_discretos + atributos_continuos]

    objetivo = titanic['Survived']
    objetivo.head()  # objetivo es una Series unidimensional

    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

    codificador_atributos_discretos = OrdinalEncoder()
    codificador_atributos_discretos.fit(atributos[atributos_discretos])

    print('Número de atributos detectados:',
        f'{codificador_atributos_discretos.n_features_in_}')
    print()
    print('Nombres de los atributos detectados:')
    print(f'{codificador_atributos_discretos.feature_names_in_}')
    print()
    print('Categorías detectadas de cada atributo:')
    for atributo, categorías in zip(
        codificador_atributos_discretos.feature_names_in_,
        codificador_atributos_discretos.categories_):
        print(f'{atributo}: {categorías}')


    atributos[atributos_discretos] = codificador_atributos_discretos.transform(atributos[atributos_discretos])

    atributos.head()


    codificador_objetivo = LabelEncoder()
    # El método fit_transform ajusta el codificador a los datos y, a continuación,
    # codifica estos adecuadamente. En este caso no necesitamos mantener el
    # atributo objetivo como una Series de Pandas.
    objetivo = codificador_objetivo.fit_transform(objetivo)
    print(f'Clases detectadas: {codificador_objetivo.classes_}')

    from sklearn.preprocessing import MinMaxScaler

    normalizador = MinMaxScaler(
        # Cada atributo se normaliza al intervalo [0, 1]
        feature_range=(0, 1)
    )

    # Como nos interesa conservar los atributos originales, realizamos la
    # normalización sobre una copia del DataFrame de atributos
    atributos_normalizados = atributos.copy()
    atributos_normalizados[:] = normalizador.fit_transform(atributos_normalizados)
    atributos_normalizados.head()

    titanic = atributos_normalizados.copy()
    titanic.head()

    
    bsa.backward_sequential_search(titanic, 'Survived', model)
