import grpc
import ping_pong_pb2_grpc as pb2_grpc
import ping_pong_pb2 as pb2
import time

class PingPongClient(object):
    def __init__(self):
        self.server_host = 'localhost'
        self.server_port = 50051
        self.channel     = grpc.insecure_channel(f"{self.server_host}:{self.server_port}")
        self.stub        = pb2_grpc.PingPongStub(self.channel)
    
    def get_ping_response(self, message):
        message = pb2.Ping(message = message)
        return self.stub.GetServerResponse(message)

if __name__ == '__main__':
    client = PingPongClient()
    message = "Ping!"

    while True:
        t = time.time()
        print(f"Sending -> {message} {t}")
        response = client.get_ping_response(message)

        print(f"Receiving -> {response.message} {response.timestamp}")
        print(f"Latency Ping -> Pong: {time.time() - t}")
        print("------------------------------------------")
        time.sleep(1)