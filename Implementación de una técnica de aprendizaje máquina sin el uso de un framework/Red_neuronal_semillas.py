#Importamos las librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargamos los datos de un archivo de texto proporcionado por kaggle
seeds = pd.read_csv("seeds_dataset.txt", header=None, sep="\t")
# Le ponemos etiquetas a las columnas 
seeds.columns = ["area", "ambiente", "compacidad", "longitud_del_nucleo", 
                 "ancho_nucleo", "coeficiente_asimetrico", "longitud_del_agujero_del_nucleo", 
                 "tipo_de_trigo"]

# Separar datos en X (características de las semillas) y Y (etiquetas el valor a clasificar)
datos_X = seeds.drop(columns=['tipo_de_trigo']).values
datos_Y = seeds['tipo_de_trigo'].values

# Escalamos los datos característicos
datos_X = (datos_X - np.mean(datos_X, axis=0)) / np.std(datos_X, axis=0)

# Dividimos los datos en entrenamiento (70%), validación (15%) y prueba (15%)
np.random.seed(42) #semilla para que de los mismos resultados
train_size = int(0.7 * datos_X.shape[0]) # datos en entrenamiento (70%)
val_size = int(0.15 * datos_X.shape[0]) # datos de validación (15%)

# Ramdomiza los indices para los datos de entrenamiento, validación y prueba
indices = np.random.permutation(datos_X.shape[0]) 
train_indices = indices[:train_size] # indices de entrenamiento
val_indices = indices[train_size:train_size+val_size] # indices de validación
test_indices = indices[train_size+val_size:] # indices de prueba

# Se obtienen los datos en los indices y se separan
X_train, Y_train = datos_X[train_indices], datos_Y[train_indices]
X_val, Y_val = datos_X[val_indices], datos_Y[val_indices]
X_test, Y_test = datos_X[test_indices], datos_Y[test_indices]

# Se crea una clase de red neuronal 
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos

        #Pesos entre la capa de entrada y la capa oculta
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        #Pesos entre la capa oculta y la capa de salida
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        # Función de activación Sigmoid. Convierte la entrada en un valor entre 0 y 1.
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        # Derivada de la función Sigmoid, usada en backpropagation
        return z * (1 - z)
    
    def forward(self, X):
        # Forward Propagation 
        # Calcula la salida de la capa oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        # Calcula la salida de la capa de salida
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2 # Retorna la salida final de la red
    
    def backward(self, X, Y, output, learning_rate):
        # Backward Propagation
        # Calcula el error entre la salida real y la predicción 
        error = output - Y
        # Gradiente de la capa de salida
        d_z2 = error * self.sigmoid_derivative(output)
        # Derivada de los pesos y sesgos de la capa de salida
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        # Gradiente de la capa oculta
        d_z1 = np.dot(d_z2, self.W2.T) * self.sigmoid_derivative(self.a1)
        # Derivada de los pesos y sesgos de la capa oculta
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Actualización de los pesos y sesgos utilizando el gradiente y la tasa de aprendizaje
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
    
    def train(self, X_train, Y_train, X_val, Y_val, epochs, learning_rate):
        # Entrenamiento de la red neuronal a lo largo de varias épocas
        for epoch in range(epochs):
            # Paso de forward propagation
            output = self.forward(X_train)
            # Paso de backward propagation y actualización de parámetros
            self.backward(X_train, Y_train, output, learning_rate)
            
            # Cada 100 épocas, calcula y muestra las pérdidas (losses) de entrenamiento y validación
            if epoch % 100 == 0:
                train_loss = np.mean(np.square(Y_train - output))
                val_output = self.forward(X_val)
                val_loss = np.mean(np.square(Y_val - val_output))
                print(f'Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
    
    def predict(self, X):
        # Realiza predicciones sobre nuevos datos, usando la red entrenada
        output = self.forward(X)
        # Redondea las salidas para obtener clases discretas
        return np.round(output)

# Convertir etiquetas en one-hot encoding
Y_train_one_hot = np.eye(3)[Y_train - 1]
Y_val_one_hot = np.eye(3)[Y_val - 1]
Y_test_one_hot = np.eye(3)[Y_test - 1]

# Inicializar y entrenar la red neuronal
input_size = X_train.shape[1] # características
hidden_size = 10  # Número de neuronas en la capa oculta
output_size = 3   # Tres tipos de trigo

nn = NeuralNetwork(input_size, hidden_size, output_size) # se crea la Red neuronal
# se entrena con los valores de entrenamiento y validación
nn.train(X_train, Y_train_one_hot, X_val, Y_val_one_hot, epochs=1000, learning_rate=0.01) 
# Evaluar con el conjunto de prueba
predictions = nn.predict(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == (Y_test - 1))
print(f'Test Accuracy: {accuracy * 100}%')

# Convertir de one-hot encoding a etiquetas
predictions_labels = np.argmax(predictions, axis=1) + 1  

# Calcular la matriz de confusión
num_classes = len(np.unique(Y_test))
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, predicted_label in zip(Y_test, predictions_labels):
    confusion_matrix[true_label-1, predicted_label-1] += 1

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Añadir las etiquetas de los ejes
classes = np.arange(1, num_classes+1)
plt.xticks(classes - 1, classes)
plt.yticks(classes - 1, classes)

# Añadir valores a cada celda
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

plt.show()
