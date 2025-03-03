import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos
data = pd.read_csv('house_prices.csv')

# Seleccionar las características y el objetivo
X = data[['num_rooms', 'size', 'location_score']]
y = data['price']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# Función para predecir el precio de una casa nueva
def predict_price(num_rooms, size, location_score):
    return model.predict([[num_rooms, size, location_score]])[0]

# Ejemplo de uso
print(predict_price(3, 120, 8))