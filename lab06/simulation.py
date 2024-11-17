
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import grpc
from concurrent import futures
import siamese_pb2_grpc as pb2_grpc
import siamese_pb2 as pb2

class ModelServer(pb2_grpc.SiameseNetworkServicer):
    def VectorRequest(self, request, context):
        id = request.id
        label = request.label
        v1 = np.frombuffer(request.v1, dtype=np.float32)

        d = self.__calculateEuclidianDistance(v1, v1)
        distance_loss = {
            "distance": d,
            "loss": self.__calculateLoss(label, d)
        }
        print("Sending back distance")
        return pb2.DistanceResponse(distance=distance_loss)

    def __calculateLoss(self, y_true, distance, margin=1.0):
        return (1 - y_true) * np.square(distance) + y_true * np.square(np.maximum(margin - distance, 0))

    def __calculateEuclidianDistance(self, v1, v2):
        result = tf.reduce_sum(tf.square(v1 - v2), axis=0)
        return tf.sqrt(tf.maximum(result, tf.keras.backend.epsilon()))


class ModelClient(object):
    def __init__(self, id):
        self.id = id
        self.server_host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.server_host}:{self.server_port}")
        self.stub = pb2_grpc.SiameseNetworkStub(self.channel)

        input = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        self.model = tf.keras.Model(inputs=input, outputs=x)

    def start(self):
        print(f"Training started for client {id}")

    def getImageOutput(self, img):
        img = np.expand_dims(img, axis=0)
        vector = self.model.predict(img)
        return vector.flatten()

    def sendToServer(self, output, label):
        request = pb2.VectorRequest(
            output,
            label,
            self.id
        )

        response = self.stub.SendOutput(request)

class Simulation:
    def __init__(self, server, client1, client2):
        self.server = server
        self.client1 = client1
        self.client2 = client2

    def __createPairs(self, x, y, num_classes):
        pairs = []
        labels = []
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
        for d in range(num_classes):
            same_class_indices = digit_indices[d]
            for i in range(len(same_class_indices)):
                # Similar Pairs
                img1 = x[same_class_indices[i]]
                pair_idx = np.random.choice(same_class_indices)
                img2 = x[pair_idx]
                pairs.append([img1, img2])
                labels.append(1)
                
                # Different Pairs
                diff_class = (d + np.random.randint(1, num_classes)) % num_classes
                diff_class_indices = digit_indices[diff_class]
                diff_idx = np.random.choice(diff_class_indices)
                img2 = x[diff_idx]
                pairs.append([img1, img2])
                labels.append(0)
                
        return np.array(pairs), np.array(labels)
    
    def __getDataset(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        return x_train, y_train, x_test, y_test

    async def start(self):
        print(f"Starting the simulation...")

        x_train, y_train, x_test, y_test = self.__getDataset()

        pairs_train, labels_train = self.__createPairs(x_train, y_train, 10)
        pairs_test, labels_test = self.__createPairs(x_test, y_test, 10)

        # Starting Server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        pb2_grpc.add_SiameseNetworkServicer_to_server(self.server, server)
        host = "[::]"
        port = 50051
        server.add_insecure_port(f"{host}:{port}")
        server.start()

        print(f"Server started at {host}:{port}")
        await server.wait_for_termination()

        # Starting Clients
        self.client1.start()
        self.client2.start()

        o1 = self.client1.getImageOutput()
        o2 = self.client2.getImageOutput()
        

if __name__ == "__main__":
    server = ModelServer()
    c1 = ModelClient(1)
    c2 = ModelClient(2)
    simulation = Simulation(server, c1, c2)
    simulation.start()