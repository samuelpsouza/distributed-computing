import grpc
from concurrent import futures

import training_pb2_grpc as pb2_grpc
import training_pb2 as pb2

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

class ModelServer(pb2_grpc.CentralizedMLPServicer):
    def GetTrainedModel(self, request, context):
        in_data = []
        in_labels = []

        for r in request.rows:
            in_data.append(r.attributes)
            in_labels.append(r.label)

        model.fit(in_data, in_labels)

        accuracy = model.score(in_data, in_labels)

        res = {
            'accuracy': accuracy
        }

        return pb2.Response(**res)
    
    def GetPrediction(self, request, context):
        ac = model.predict(request.row)

        res = {
            'accuracy': ac
        }

        return pb2.Response(**res)
    
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