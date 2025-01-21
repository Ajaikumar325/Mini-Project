import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D, Flatten, Dense, Activation, Add
from tensorflow.keras.optimizers import Adam

# Assuming you have your data loaded as X_train and y_train
# Now let's integrate the ResNet architecture and feature selection with a smaller sample file
df = pd.read_csv("new.csv")

# Encode categorical labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Separate features and target variable
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Add a time step dimension
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

class BGOA:
    def __init__(self, X_train, y_train, X_test, y_test, population_size=20, max_iter=25):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_features = X_train.shape[1]
class BGOA:
    def __init__(self, X_train, y_train, X_test, y_test, population_size=20, max_iter=25):
        self.num_features = X_train.shape[1]
        self.population = np.random.randint(2, size=(population_size, self.num_features))
        self.fitness = np.zeros(population_size)

    def fitness_function(self, solution):
        selected_features = [bool(s) for s in solution]
        X_selected = self.X_train[:, selected_features]

        # Build and compile ResNet model
        self.model = self.build_resnet(X_selected.shape[1:])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train ResNet model
        self.model.fit(X_selected, self.y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate accuracy
        _, accuracy = self.model.evaluate(X_selected, self.y_train, verbose=0)

        return accuracy

    def levy_flight(self):
        beta = 1.5
        sigma1 = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                  (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=self.num_features)
        v = np.random.normal(0, sigma2, size=self.num_features)
        step = u / (np.abs(v) ** (1 / beta))
        return step
class BGOA:
    def residual_block(self, x, filters, strides=2):
        shortcut = x
        x = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv1D(filters * 4, kernel_size=1, strides=1, padding='same')(x)
        shortcut = Conv1D(filters * 4, kernel_size=1, strides=strides, padding='same')(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

# Instantiate the feature selector
selector = BGOA(X_train, y_train, X_test, y_test)

# Select features
selected_features = selector.select_features()

print("Selected Features:", selected_features)

selected_features = [feature for feature, selected in zip(df.columns[:-1], selected_features) if selected == 1]

# Load the CICIDS 2017 dataset
dataset = pd.read_csv("Concatenated_dataset.csv")

dataset = dataset[selected_features + ['Label']]

# Encode categorical labels after selecting features
label_encoder = LabelEncoder()
dataset['Label'] = label_encoder.fit_transform(dataset['Label'])

# Separate features and target variable
X = dataset.drop(columns=['Label']).values
y = dataset['Label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to add a time step dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
def build_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=2, padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D()(x)

    # Block1
    x = residual_block(x, filters=32, strides=1)
    # Block2
    x = residual_block(x, filters=64, strides=2)
    # Block3
    x = residual_block(x, filters=128, strides=2)
    # Block4
    x = residual_block(x, filters=256, strides=2)

    x = AveragePooling1D(pool_size=1)(x)
    x = Flatten()(x)
    x = Dense(40, activation='tanh')(x)
    outputs = Dense(len(np.unique(y_train)), activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def residual_block(x, filters, strides=2):
    shortcut = x
    x = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters * 2, kernel_size=3, strides=1, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters * 4, kernel_size=1, strides=1, padding='same')(x)
    shortcut = Conv1D(filters * 4, kernel_size=1, strides=strides, padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
