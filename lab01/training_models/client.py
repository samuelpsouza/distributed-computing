import grpc
import training_pb2 as pb2
import training_pb2_grpc as pb2_grpc

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class ModelClient(object):
    def __init__(self):
        self.server_host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.server_host}:{self.server_port}")
        self.stub = pb2_grpc.CentralizedMLPStub(self.channel)
    
    def getModelAccuracy(self, message):
        return self.stub.GetTrainedModel(message)
    
    def getModelPrediction(self, message):
        return self.stub.GetPrediction(message)

if __name__ == '__main__':
    iris = load_iris()
    attributes = iris.data
    labels     = iris.target

    x_training, x_test, y_training, y_test = train_test_split(attributes, labels, test_size = 0.2)

    fit_req = pb2.FitRequest()

    for index in range(len(x_training)):
        row = pb2.Row()
        row.attributes.extend(x_training[index])
        row.label = y_training[index]
        fit_req.rows.append(row)
    
    client = ModelClient()
    answer = client.getModelAccuracy(fit_req)

    print(f"Training : {answer}")

    for index in range(len(x_test)):
        pred_req = pb2.PredictRequest()
        pred_req.row.attributes.extend(x_test[index])

        print(f"{index} -> Prediction -> {client.getModelPrediction(pred_req)}")


