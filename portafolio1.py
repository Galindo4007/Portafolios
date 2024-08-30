import numpy as np # Se importa numpy por ciertas funciones de multiplicaciones de matrices
import pandas as pd # Pandas se usa para leer los datos del csv de powerlifting
import matplotlib.pyplot as plt # Este se usa para graficar los datos de la regresión lineal y los puntos predichos
from sklearn.preprocessing import StandardScaler # Esto se usa para escalar los datos

# En la primera parte se lee los datos del archivo
powerdata = pd.read_csv('openpowerlifting.csv')
# Fin de la lectura de los datos

# Se asignan las columnas correspondientes para poder hacer la asignación de features, instances y labels
powerdata.columns = ["MeetID", "Name", "Sex", "Equipment", "Age", "Division", "BodyweightKg",
                     "WeightClassKg", "Squat4Kg", "BestSquatKg", "Bench4Kg", "BestBenchKg",
                     "Deadlift4Kg", "BestDeadliftKg", "TotalKg", "Place", "Wilks"]
# Fin de la asignación de las columnas

# Hay varios datos en los levantamientos de los básicos que son negativos, por lo que se decidió eleminarlos
# De la manera que se hizo fue buscar el df todos los que fueran mayores o iguales a 0
df = powerdata[(powerdata['BestBenchKg'] >= 0) &
               (powerdata['BestSquatKg'] >= 0) &
               (powerdata['BestDeadliftKg'] >= 0)]
# Eliminar solo los NaNs en las columnas especificadas
df = df.dropna(subset=['BestBenchKg', 'BestSquatKg', 'BestDeadliftKg', "BodyweightKg"])
# Fin de la limpieza de los datos

# Con propósito de análisis se muestra la información correspondiente de toda la tabla de varias partes de la columna
# muestra la cantidad de datos, el mayor dato, la desviación estandar entre otras cosas
print(df.describe())
# Fin de la descripción

# Asignación de los Features, instances y labels
X_powerdata = df[["BestSquatKg", "BestBenchKg", "BestDeadliftKg"]]
Y_powerdata = df["BodyweightKg"]
# Fin de la asignación para el calculo

# Muestra de los datos y la forma de los datos
print("Feature y sample Shape: ", X_powerdata.shape)
print("Label Shape: ", Y_powerdata.shape)
print(X_powerdata.head())
print(Y_powerdata.head())
# Fin de la muestra de datos

# Se calcula el tamaño de Xpowerdata para agarrar el 20% de los datos y poder dividirlos
tam = len(X_powerdata)
tam2 = int(tam * 0.2)  # Tamaño del conjunto de prueba

# Crear un array de índices del tamaño del arreglo de Xpowerdata
indices = np.arange(tam)

# Mezclar los índices aleatoriamente
np.random.shuffle(indices)

# Dividir los índices en entrenamiento y prueba
# Esto se hace de manera en la que los indices están mezclados y
# los indices del 0 al tam2 que esto son los datos de prueba y son el 20% y en la parte de entrenamiento
# son el resto de datos que empiezan desde el tam2 hasta el final lo que es el 80%.
test_indices = indices[:tam2]
train_indices = indices[tam2:]

# una vez seleccionados los datos de entrenamiento y de prueba se dividen seégun el indice tanto en los
# Features e instances como en los Labels
X_test = X_powerdata.iloc[test_indices]
Y_test = Y_powerdata.iloc[test_indices]
X_train = X_powerdata.iloc[train_indices]
Y_train = Y_powerdata.iloc[train_indices]

# Por motivos graficos se enseñan todos los datos para comprobar
# los tamaños respectivamente y que si concuerde con la infromación dada
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Se escalan los datos tanto los de prueba como los datos de entrenamiento
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Se agrega y se muestra un theta con datos aleatorios para empezar el modelo
theta = np.random.randn(len(X_train[0]) + 1, 1)
print("theta:", theta)

# En la parte del Entrenamiento se agrega una fila de unos para los consecuentes calculos
# y se muestra la forma de los datos
X_vect_powerdata = np.c_[np.ones((len(X_train), 1)), X_train]
print("X_vect_powerdata: ", X_vect_powerdata[:5])
print("X_vect_powerdata shape: ", X_vect_powerdata.shape)

# En la parte del Prueba se agrega una fila de unos para los consecuentes calculos
# y se muestra la forma de los datos
X_test_vect = np.c_[np.ones((len(X_test), 1)), X_test]
print("X_test_vect: ", X_test_vect[:5])
print("X_test_vect shape: ", X_test_vect.shape)

# Se crea una función de regresión lineal
def regresion_lineal(X, y, theta, alpha, num_iters):
    y_ = np.reshape(y, (len(y), 1))  # Asegúrate de que y sea una columna
    N = len(X) # se obtiene el tamaño de N para los calculos de abajo
    avg_loss_list = [] # se crea una lista de porcentaje pérdido
    prev_cost = float('inf')  # Inicializa el costo anterior con infinito

    # Se emepiezan iteraciones, según lo que hayamos elegido
    for i in range(num_iters):
        h = np.dot(X, theta) # Se hace la predicción de los datos con las thetas y X respectivamente
        error = h - y_ # Luego de esto se verifica con los datos de entrenamiento para sacar el error
        gradiente = (1 / N) * np.dot(X.T, error) # Con el calculo del error se puede sacar el gradiente
        # ya que solo necesitamos de la traspuesta de X y multiplicarla con el error y
        # eso dividirlo entre la cantidad de datos de X
        theta = theta - alpha * gradiente # Se actualiza la theta, a través del rango de aprendizaje
        costo = (1 / (2 * N)) * np.sum(error**2) # Se calcula el costo de la iteración
        print(f"Costo de la iteración: {costo}, número de iteración: {i}") # Se muestra la información
        # del entrenamiento

        # Se agrega el porcentaje de pérdida a la lista
        avg_loss_list.append(costo)
        # Verificar la condición de parada esto se hace para que no necesariamente tenag que terminar con
        # el ciclo de vueltas en una canidad donde el error es muy mínimo, ya que compara el costo anterior
        # con el costo actual y se cumple se regresan los datos
        if abs(prev_cost - costo) < 0.001:
            print(f"Convergencia alcanzada en la iteración {i}")
            return theta, avg_loss_list
        # Se actualiza el costo anterior para la siguiente iteración
        prev_cost = costo

    return theta, avg_loss_list # Se regresan los datos si se alcanza el maximo de iteraciones
    # Fin de la función de regresión lineal

# Se llama a la función de regresión lienal donde recibimos ambos parámetros tanto theta como la lista del
# porcentaje de costo, pero el proceso se muestra en la función por lo que el dato que nos interesa para la predicción
# es theta
theta, avg_loss_list = regresion_lineal(X_vect_powerdata, Y_train, theta, 0.01, 1000)

# Se hace la multiplicación de matrices de los datos de prueba ya estandarizados con el theta más óptimo
# y luego se comparan los primeros 5 con los datos reales
Y_pred = np.dot(X_test_vect, theta)
print("Y_pred: ", Y_pred[:5])
print("Y_test: ", Y_test[:5])

# Función para calcular el r al cuadrado, para saber si el modelo es bueno o no, ya que en
# regresión lineal es el que nos puede indicar si nuestro modelo es bueno o no, porque en los otros
# que es el MSE se tienen que considerar otros contextos en este caso el que mejor se ajusta es el r al cuadrado
# por los tipos de datos que se estan manejando
def calcular_r2(Y_real, Y_pred):
    # Suma de los cuadrados de los residuos (errores)
    ss_res = np.sum((Y_real - Y_pred) ** 2)
    # Suma total de los cuadrados (variabilidad de los datos)
    ss_tot = np.sum((Y_real - np.mean(Y_real)) ** 2)
    # Coeficiente de determinación R²
    r2 = 1 - (ss_res / ss_tot)
    # Se regresa el r al cuadrado
    return r2
# Se aplana los datos de la Y_pred para que se puedan hacer los calculos correspondientes
Y_pred = np.ravel(Y_pred)
# Se verifica que tienen la misma forma
print("Y_test shape:", Y_test.shape)
print("Y_pred shape:", Y_pred.shape)
# Se llama a la función y se muestran los datos, la r al cuadrado suele estar entre 0 a 1.
r2 = calcular_r2(Y_test, Y_pred)
print("R2: ", r2)

# Función para graficar los resultados
def graficar_regresion_lineal(X, Y, Ypred):
    # Se  grafican los puntos de los datos de prueba y las Y de prueba para ver donde están los resultados
    plt.scatter(X[:, 1], Y, color='red', marker='o', label='Datos Reales')
    # Se grafica la regresión lineal
    plt.plot(X[:, 1], Ypred, color='blue', label='Regresión Lineal')
    # Se muestran los datos de la gráfica
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Datos Reales vs. Datos Predichos')
    plt.show()
    # Se muestra la gráfica final
# Fin de la función que grafica los resultados

# Se llama a la función que grafica los resultados
graficar_regresion_lineal(X_test_vect, Y_test, Y_pred)