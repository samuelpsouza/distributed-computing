import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf
import keras
from keras.models import Model
import grpc
import time
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_partial_model(input_layer):
    layer1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = keras.layers.MaxPooling2D((2, 2))(layer1)
    layer3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(layer2)
    layer4 = keras.layers.Dense(128, activation='relu')(layer3)
    return Model(inputs=input_layer, outputs=layer4)


def get_activations(model, X):
    with tf.GradientTape(persistent=True) as tape:
        activations = model(X)
    return activations, tape

def send_activations_to_server(stub, activations, labels, batch_size, client_id):
    activations_list = activations.numpy().flatten()

    client_to_server_msg = pb2.ClientToServer()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.labels.extend(labels.flatten())
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id = client_id

    server_response = stub.SendClientActivations(client_to_server_msg)
    return server_response

# CIFAR-10 dataset
cifar10                              = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test                      = X_train / 255.0, X_test / 255.0

partial_model    = create_partial_model(keras.layers.Input(shape=(32, 32, 3)))
client_optimizer = tf.keras.optimizers.Adam()

MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
channel = grpc.insecure_channel('localhost:50051', options=[
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
])
stub = pb2_grpc.SplitLearningStub(channel)

def train_step(model, x_batch, y_batch, batch, optimizer, epoch, stub):

    activations, tape     = get_activations(model, x_batch)
    flattened_activations = tf.reshape(activations, (activations.shape[0], -1))

    latencia_start  = time.time()
    server_response = send_activations_to_server(stub, flattened_activations, y_batch, len(x_batch), 1)
    latencia_end    = time.time()

    print("Received response from server")
    activations_grad = tf.convert_to_tensor(server_response.gradients, dtype=tf.float32)
    activations_grad = tf.reshape(activations_grad, activations.shape)

    client_gradient = tape.gradient(
        activations,
        model.trainable_variables,
        output_gradients=activations_grad
    )

    bytes_tx  = flattened_activations.numpy().nbytes
    bytes_rx  = activations_grad.numpy().nbytes
    latencia  = latencia_end - latencia_start
    loss      = server_response.loss
    acc       = server_response.acc

    print(f"Latencia: {latencia} segundos")
    print(f"Data Tx: {bytes_tx / 2**20} MB")
    print(f"Data Rx: {bytes_rx / 2**20} MB")

    optimizer.apply_gradients(zip(client_gradient, model.trainable_variables))

    with open('lab08/results.csv', 'a') as f:
        f.write(f"{epoch}, {batch}, {loss}, {acc}, {latencia}, {bytes_tx / 2**20}, {bytes_rx / 2**20}\n")

for epoch in range(10):
    batch_size = 64
    n_batches  = X_train.shape[0]//batch_size

    for batch in range(n_batches):
        print(f"Epoch {epoch} - Batch {batch}/{n_batches}")

        X_batch  = X_train[batch_size * batch : batch_size * (batch+1)]
        y_batch  = y_train[batch_size * batch : batch_size * (batch+1)]

        train_step(partial_model, X_batch, y_batch, batch, client_optimizer, epoch, stub)
