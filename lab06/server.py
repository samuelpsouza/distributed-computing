import grpc
from concurrent import futures

import training_pb2_grpc as pb2_grpc
import training_pb2 as pb2

import tensorflow as tf

class ModelServer(pb2_grpc.CentralizedMLPServicer):
    def __init__(self):
        self.model = self.createModel();

    # Create a model for Nzz
    def createModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')
        ])
        return model
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CentralizedMLPServicer_to_server(ModelServer(), server)
    host = "[::]"
    port = 50051
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"Server started at {host}:{port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()