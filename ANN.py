import tensorflow as tf
import numpy as np

# -----------------------------
# INPUT DATA
# -----------------------------

# Normalized Titanic features
# x1 = Pclass
# x2 = Age
# x3 = Fare

X = np.array([[0.20, 0.24, 0.80]], dtype=np.float32)

# Target Output
y = np.array([[1]], dtype=np.float32)

# -----------------------------
# CREATE MODEL
# -----------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)), # Explicitly define the input layer
    tf.keras.layers.Dense(
        2,
        activation='sigmoid',
        name='hidden_layer'
    ),
    
    tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        name='output_layer'
    )
])

# -----------------------------
# SET INITIAL WEIGHTS MANUALLY
# -----------------------------

# Input → Hidden weights
hidden_weights = np.array([
    [0.11, 0.21],   # x1 weights
    [0.14, 0.24],   # x2 weights
    [0.17, 0.27]    # x3 weights
], dtype=np.float32)

# Hidden biases
hidden_biases = np.array([0.1, 0.1], dtype=np.float32)

# Hidden → Output weights
output_weights = np.array([
    [0.31],
    [0.34]
], dtype=np.float32)

# Output bias
output_biases = np.array([0.1], dtype=np.float32)

# Assign weights using layer names for robustness
model.get_layer('hidden_layer').set_weights([hidden_weights, hidden_biases])
model.get_layer('output_layer').set_weights([output_weights, output_biases])

# -----------------------------
# COMPILE MODEL
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mean_squared_error'
)

# -----------------------------
# FORWARD PROPAGATION
# -----------------------------

# Hidden layer output
hidden_model = tf.keras.Model(
    inputs=model.layers[0].input, # Use the input from the first layer
    outputs=model.get_layer('hidden_layer').output
)

hidden_output = hidden_model.predict(X)

print("\nHidden Layer Outputs:")
print(hidden_output)

# Final prediction
prediction = model.predict(X)

print("\nFinal Predicted Output:")
print(prediction)

# -----------------------------
# ERROR CALCULATION
# -----------------------------

loss = model.evaluate(X, y, verbose=0)

print("\nMean Squared Error:")
print(loss)

# -----------------------------
# TRAIN FOR 1 EPOCH
# (Backpropagation + Weight Update)
# -----------------------------

model.fit(X, y, epochs=1, verbose=1)

# -----------------------------
# UPDATED WEIGHTS
# -----------------------------

print("\nUpdated Weights and Biases:\n")

# Hidden layer updated values
updated_hidden_weights, updated_hidden_biases = model.get_layer('hidden_layer').get_weights()

print("Input → Hidden Weights:")
print(updated_hidden_weights)

print("\nHidden Biases:")
print(updated_hidden_biases)

# Output layer updated values
updated_output_weights, updated_output_biases = model.get_layer('output_layer').get_weights()

print("\nHidden → Output Weights:")
print(updated_output_weights)

print("\nOutput Bias:")
print(updated_output_biases)

print("\nThis is tensorflow method")

# Save model
model.save("titanic_ann_model.h5")

print("Model Saved Successfully!")
