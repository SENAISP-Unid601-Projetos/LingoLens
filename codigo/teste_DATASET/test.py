import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Caminho para o dataset
caminho_dataset = r'C:\Users\Aluno\LingoLens\teste_DATASET'

# Configurações
img_height, img_width = 128, 128
batch_size = 32

# Carregar dataset (original para treino e validação)
treino_ds_original = tf.keras.preprocessing.image_dataset_from_directory(
    caminho_dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds_original = tf.keras.preprocessing.image_dataset_from_directory(
    caminho_dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Mostrar classes
class_names = treino_ds_original.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalização dos pixels (0–1)
normalization_layer = layers.Rescaling(1./255)
treino_ds = treino_ds_original.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_original.map(lambda x, y: (normalization_layer(x), y))

# Melhor performance com cache e prefetch
AUTOTUNE = tf.data.AUTOTUNE
treino_ds = treino_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Modelo CNN simples
modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Classificação multiclasse
])

modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # <<< aqui!
    metrics=['accuracy']
)


# Treinar
historico = modelo.fit(
    treino_ds,
    validation_data=val_ds,
    epochs=10
)

import matplotlib.pyplot as plt

acc = historico.history['accuracy']
val_acc = historico.history['val_accuracy']
loss = historico.history['loss']
val_loss = historico.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia Treino')
plt.plot(epochs_range, val_acc, label='Acurácia Validação')
plt.legend()
plt.title('Acurácia durante o Treino')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda Treino')
plt.plot(epochs_range, val_loss, label='Perda Validação')
plt.legend()
plt.title('Perda durante o Treino')

plt.show()


# Salvar modelo
modelo.save('meu_modelo')  # salva numa pasta chamada 'meu_modelo'
