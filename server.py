import socket
import csv



# Set up the server for receiving data from the client
server_ip = "0.0.0.0"   # Listen on all available interfaces
server_port = 8080  # Port to listen on 

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the IP and port
server_socket.bind((server_ip, server_port))

# Listen for incoming connections (1 connection at a time)
server_socket.listen(1)
print(f"Server started on {server_ip}:{server_port}. Waiting for Arduino connection...")

# List to keep track of connected clients
connected_clients = []

label = "trick" # esim sessio,koehenkilö,aktiviteetti 
# Open CSV file for writing data
csv_filename = "./data/Vuokko_{}.csv".format(label)

try:
    while True:
        # Accept a connection from the Arduino
        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address}")

        # Add the client socket to the list of connected clients
        connected_clients.append(client_socket)

        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Sensor1", "Sensor2", "Sensor3", "Sensor4"])  # Write header

            try:
                while True:
                    # Receive data from the Arduino
                    data = client_socket.recv(1024).decode("utf-8").strip() # Reads maximum of 1024 bytes
                    if not data:
                        # print("No data received.")
                        continue
                    else:
                        # Parse the CSV data
                        parts = data.split(",")
                        if len(parts) == 5:  # Ensure the packet has at least Timestamp, and one sensor value
                            timestamp = parts[0]
                            sensor1 = parts[1]  # Remaining parts are sensor values
                            sensor2 = parts[2]
                            sensor3 = parts[3]
                            sensor4 = parts[4]

                            # Print the received data
                            print(f"Received timestamp={timestamp}")
                            # Save to CSV
                            writer.writerow([timestamp, sensor1, sensor2, sensor3, sensor4])
                            file.flush()  # Ensure data is written immediately
                        
            except ConnectionResetError:
                print(f"Connection to {client_address} reset by client.")

            finally:
                # Remove the client from the list of connected clients
                connected_clients.remove(client_socket)
                client_socket.close()
                print(f"Connection to {client_address} closed.")

except KeyboardInterrupt:
    # Handle Ctrl+C to stop the server gracefully
    print("Server stopped.")
finally:
    # Close all client sockets
    for client in connected_clients:
        client.close()
    # Close the server socket
    server_socket.close()