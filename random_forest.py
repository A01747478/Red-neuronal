import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sn

# Cargar dataset de dígitos
digits = load_digits()
df = pd.DataFrame(digits.data)
df["Target"] = digits.target  # Añadir columna de target

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(["Target"], axis="columns"))

# Dividir los datos en entrenamiento (60%), validación (20%) y prueba (20%)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X_scaled, digits.target, test_size=0.2, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

# Ajustar los hiperparámetros para reducir aún más el sobreajuste
param_grid = {
    'n_estimators': [50, 100],                # Aumentar el número de estimadores
    'criterion': ['gini'],                    
    'max_depth': [3, 5],                      # Limitar más la profundidad del árbol
    'min_samples_split': [10, 15],            # Aumentar el número mínimo de muestras por división
    'min_samples_leaf': [4, 6],               # Aumentar el número mínimo de muestras en las hojas
    'max_features': ['sqrt', 'log2', 0.5],    # Limitar más las características por split
    'max_samples': [0.7, 0.8]                 # Utilizar una fracción de muestras para cada árbol
}

# Realizar la búsqueda de hiperparámetros
grid_search = GridSearchCV(RandomForestClassifier(bootstrap=True), param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Usar el mejor modelo encontrado
best_model = grid_search.best_estimator_
print(f"Precisión en validación: {best_model.score(X_val, Y_val)}")

# Realizar predicciones y reporte de clasificación para el conjunto de validación
y_val_predicted = best_model.predict(X_val)
print("Reporte de Clasificación para el Conjunto de Validación:\n", classification_report(Y_val, y_val_predicted))

# Visualizar la matriz de confusión para el conjunto de validación
cm_val = confusion_matrix(Y_val, y_val_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión para el Conjunto de Validación')
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos')
plt.show()

# Generar curva de aprendizaje usando los datos de entrenamiento y prueba
train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model,
    X=X_train_full,  # Usar todo el conjunto de entrenamiento completo para mejor generalización
    y=Y_train_full,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,  # Aumentar cv para más validación cruzada
    scoring='accuracy',
    shuffle=True
)

# Calcular las medias y desviaciones estándar
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Visualizar la curva de aprendizaje
plt.figure(figsize=(8, 5))
plt.title("Curva de Aprendizaje")
plt.xlabel("Tamaño de Entrenamiento")
plt.ylabel("Precisión")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntaje de entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Puntaje de prueba")
plt.legend(loc="best")
plt.show()

# Evaluar en el conjunto de prueba y mostrar el reporte de clasificación
y_predicted = best_model.predict(X_test)
cm_test = confusion_matrix(Y_test, y_predicted)
print(f"Precisión en el Conjunto de Prueba: {best_model.score(X_test, Y_test)}")

print("Reporte de Clasificación para el Conjunto de Prueba:\n", classification_report(Y_test, y_predicted))

# Visualizar la matriz de confusión para el conjunto de prueba
plt.figure(figsize=(10,7))
sn.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión para el Conjunto de Prueba')
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos')
plt.show()
