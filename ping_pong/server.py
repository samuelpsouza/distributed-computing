import grpc
from concurrent import futures
import time

import ping_pong_pb2_grpc as pb2_grpc
import ping_pong_pb2 as pb2

class PingPongServer(pb2_grpc.PingPongServicer):
    def GetServerResponse(self, request, context):
        message = request.message
        print(f"received message: {message}")

        response = f"Pong!"

        reponse_message = {
            'message': response,
            'timestamp': time.time()
        }

        return pb2.Pong(**reponse_message)

def bootstrap():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_PingPongServicer_to_server(PingPongServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print("Server started at 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    bootstrap()