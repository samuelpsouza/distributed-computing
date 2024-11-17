import grpc
import siamese_pb2 as pb2
import siamese_pb2_grpc as pb2_grpc
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

class ModelClient(object):
    def __init__(self):
        self.id = 1
        self.server_host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.server_host}:{self.server_port}")
        self.stub = pb2_grpc.SiameseNetworkStub(self.channel)


    def start(self):
        print(f"Training started for client {self.id}")

    # def getImageOutput(self, img):
    #     img = np.expand_dims(img, axis=0)
    #     vector = self.model.predict(img)
    #     return vector.flatten()

    def sendToServer(self, message):
        return self.stub.SendVectors(message)

def __createPairs(x, y, num_classes):
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
    
def __getDataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    print(f"Starting the simulation...")

    x_train, y_train, x_test, y_test = __getDataset()

    pairs_train, labels_train = __createPairs(x_train, y_train, 10)
    pairs_test, labels_test = __createPairs(x_test, y_test, 10)
    client = ModelClient()
    #o1 = client.getImageOutput(pairs_train[0][0])
    fit_req = pb2.VectorRequest()
    fit_req.v1 = 1
    answer = client.sendToServer(fit_req)