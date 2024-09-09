import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Definir hiperparámetros para GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20]
}

# Optimización de hiperparámetros con el conjunto de validación
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, Y_train)

# Evaluar el modelo en el conjunto de validación
print(f"Precisión en validación: {grid_search.score(X_val, Y_val)}")

# Usar el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Entrenar nuevamente el mejor modelo con datos de entrenamiento y validación combinados
X_train_combined = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)])
Y_train_combined = pd.concat([pd.Series(Y_train), pd.Series(Y_val)])

best_model.fit(X_train_combined, Y_train_combined)

# Evaluar el modelo final en el conjunto de prueba
y_predicted = best_model.predict(X_test)

print(f"Precisión en prueba: {best_model.score(X_test, Y_test)}")

# Matriz de confusión
cm = confusion_matrix(Y_test, y_predicted)

# Reporte de clasificación
print("Reporte de clasificación:\n", classification_report(Y_test, y_predicted))

# Visualizar la matriz de confusión
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.title("Matriz de Confusión")
plt.show()
