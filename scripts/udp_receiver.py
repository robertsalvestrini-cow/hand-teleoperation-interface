import json
import socket


def main():
    host = "127.0.0.1"
    port = 5005

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print(f"Listening on UDP {host}:{port}")

    while True:
        data, addr = sock.recvfrom(4096)
        payload = json.loads(data.decode("utf-8"))
        print(payload)
if __name__ == "__main__":
    main()