import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf
import keras
from keras.models import Model
import grpc
import time
import numpy as np
import os
import ray

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_partial_model(input_layer):
    layer1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = keras.layers.MaxPooling2D((2, 2))(layer1)
    layer3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(layer2)
    layer4 = keras.layers.Flatten()(layer3)
    layer5 = keras.layers.Dense(128, activation='relu')(layer4)
    return Model(inputs=input_layer, outputs=layer5)

def create_client_final_model(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)
    output_layer = keras.layers.Dense(10, activation='softmax')(input_layer)
    return Model(inputs=input_layer, outputs=output_layer)

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

# MNIST dataset
#mnist = keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test                      = X_train / 255.0, X_test / 255.0

partial_m1 = create_partial_model(keras.layers.Input(shape=(32, 32, 3)))
partial_m3 = create_client_final_model((128,))

client_optimizer = tf.keras.optimizers.Adam()

def train_step(m1, m3, x_batch, y_batch, optimizer, epoch):
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    channel = grpc.insecure_channel('172.17.0.1:50051', options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])

    stub = pb2_grpc.SplitLearningStub(channel)

    activations, tape     = get_activations(m1, x_batch)
    flattened_activations = tf.reshape(activations, (activations.shape[0], -1))

    latencia_start  = time.time()
    server_response = send_activations_to_server(stub, flattened_activations, y_batch, len(x_batch), 1)
    latencia_end    = time.time()

    # Inicio de backpropagation
    activations_grad = tf.convert_to_tensor(server_response.gradients, dtype=tf.float32)
    activations_grad = tf.reshape(activations_grad, activations.shape)

    # Atualizando M3
    with tf.GradientTape() as tape_m3:
        predictions_m3 = m3(activations)  # Forward pass through M3
        loss_m3 = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions_m3)
        loss_m3 = tf.reduce_mean(loss_m3)

    gradients_m3 = tape_m3.gradient(loss_m3, m3.trainable_variables)
    optimizer.apply_gradients(zip(gradients_m3, m3.trainable_variables))

    # Atualizando M1
    client_gradient = tape.gradient(
        activations,
        m1.trainable_variables,
        output_gradients=activations_grad
    )

    optimizer = tf.keras.optimizers.Adam()
    optimizer.apply_gradients(zip(client_gradient, m1.trainable_variables))

    bytes_tx  = flattened_activations.numpy().nbytes
    bytes_rx  = activations_grad.numpy().nbytes
    latencia  = latencia_end - latencia_start
    loss      = server_response.loss
    acc       = server_response.acc

    print(f"Latencia: {latencia} segundos")
    print(f"Data Tx: {bytes_tx / 2**20} MB")
    print(f"Data Rx: {bytes_rx / 2**20} MB")
    
    with open(f'/tmp/results.csv', 'a') as f:
        f.write(f"{epoch}, {loss}, {acc}, {latencia}, {bytes_tx / 2**20}, {bytes_rx / 2**20}\n")

ray.init(
    address="ray://localhost:10001",
    runtime_env={
        "working_dir": "/home/samuel/Coding/distributed-computing/lab08/exercise",
        "pip": ["tensorflow", "keras", "grpcio", "numpy"]})

@ray.remote
def training_client(m1, m3, x_batch, y_batch, optimizer, epoch):
    train_step(m1, m3, x_batch, y_batch, optimizer, epoch)
        

num_clients = 10
batch_size = 16
#client_data_size = X_train.shape[0] // num_clients
client_data_size = 1000 // num_clients
#n_batches  = X_train.shape[0]//batch_size
n_batches  = 1000 // batch_size

clients = []

for epoch in range(10):
    for client_id in range(num_clients):
        X_batch = X_train[client_id * client_data_size: (client_id + 1) * client_data_size]
        y_batch = y_train[client_id * client_data_size: (client_id + 1) * client_data_size]
        client_task = training_client.remote(partial_m1, partial_m3, X_batch, y_batch, client_optimizer, epoch)
        clients.append(client_task)
        

ray.get(clients)
