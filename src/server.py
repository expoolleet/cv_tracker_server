import os
import threading
import socket
import json
import netifaces
import asyncio

from src.event import post_event
from src.event_types import UPDATE_TRACKING, STOP_TRACKING, START_STREAM_FOR_CLIENT, STOP_STREAM_FOR_CLIENT, SEND_CFS, SHOW_CAMERA_PREVIEW, STOP_CAMERA_PREVIEW
from src.command import Command

SOCKET_RECEIVE_BUFFER = 1024

def get_ip_for_interface(interface_name):
    addresses = netifaces.ifaddresses(interface_name)
    ip_info = addresses[netifaces.AF_INET][0]
    return ip_info['addr']


class Server:
    def __init__(self, interface_name, port):
        self.ip = get_ip_for_interface(interface_name)
        self.port = port
        self.client_connections = []
        self.client_connections_lock = threading.Lock()
        self.clients = {}
        threading.Thread(target=self.start_socket_server, daemon=True).start()
    
    
    def start_socket_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reusing the address
            s.bind((self.ip, self.port))
            s.listen()
            print(f"Socket server listening on {self.ip}:{self.port}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()


    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        with self.client_connections_lock:
            self.client_connections.append(conn)
        try:
            while True:
                data = conn.recv(SOCKET_RECEIVE_BUFFER)
                if not data:
                    break
                try:    
                    message = json.loads(data.decode('utf-8'))
                    self.handle_message(message, addr)
                except json.JSONDecodeError:
                    print(f"Received non-JSON data: {data.decode('utf-8').strip()}")
        except Exception as e:
            print(f"Error in client handler: {e}")
        finally:
            print(f"Client {addr} disconnected")
            with self.client_connections_lock:
                self.client_connections.remove(conn)
            conn.close() 
            
            
    def handle_message(self, message, client_addr):
        print(f"Client message {message}")
        if message["command"] == Command.START_STREAM:
            post_event(START_STREAM_FOR_CLIENT, 
                       {
                            "ip": client_addr[0],
                            "stream_size": message["data"]["stream_size"],
                            "bitrate": message["data"]["bitrate"],
                            "frame_rate": message["data"]["frame_rate"]
                       })
        elif message["command"] == Command.STOP_STREAM:
            post_event(STOP_STREAM_FOR_CLIENT, client_addr[0])
        elif message["command"] == Command.UPDATE_TRACKING:
            post_event(UPDATE_TRACKING, message["data"])
        elif message["command"] == Command.STOP_TRACKING:
            post_event(STOP_TRACKING)
        elif message["command"] == Command.SEND_CFS:
            post_event(SEND_CFS, message["data"])
        elif message["command"] == Command.START_TRANSMISSION:
            post_event(SHOW_CAMERA_PREVIEW,
                       {    
                            "ip": client_addr[0],
                            "frame_rate": message["data"]["frame_rate"]
                       })
        elif message["command"] == Command.STOP_TRANSMISSION:
            post_event(STOP_CAMERA_PREVIEW, client_addr[0])
        elif message["command"] == Command.REBOOT_SERVER:
            self.reboot()
        else:
            print(f"Received unknown message: {message}")
            
    
    def send_command_to_clients(self, command, data=None):
        try:
            message = json.dumps({"command": command, "data": data if data else ''}).encode('utf-8') + b'\n'
            with self.client_connections_lock:
                for conn in self.client_connections:
                    conn.sendall(message)
        except BrokenPipeError:
            pass  # Client disconnected, ignore this error
        except Exception as e:
            print(f"Error sending command {command} with data {data} to clients: {e}")            

            
    def send_new_roi_to_clients(self, roi):
            data = json.dumps({"roi": list(roi)}).encode('utf-8') + b'\n'
            with self.client_connections_lock:
                try:
                    for conn in self.client_connections:
                        conn.sendall(data)
                except Exception as e:
                    print(f"Error sending ROI to clients: {e}")
        
                    
    async def _send_new_roi_to_clients_async(self, conn, roi_data, delay):
        await asyncio.sleep(delay)
        try:    
            conn.sendall(roi_data)
        except Exception as e:
                print(f"Error sending ROI to clients: {e}")
        
        
    def send_new_roi_to_clients_async(self, roi, delay=0.25):
        roi_data = json.dumps({"roi": list(roi)}).encode('utf-8')
        with self.client_connections_lock:   
            for conn in self.client_connections:
                asyncio.create_task(self._send_new_roi_to_clients_async(conn, roi_data, delay))
     
    
    def clear_client(self, ip):
        with self.client_connections_lock:
            if ip in self.clients:
                del self.clients[ip]
                print(f"Client {ip} cleared from server.")
            else:
                print(f"Client {ip} not found in server clients.")
    
    
    def add_pipe_to_client(self, ip, pipe):
        with self.client_connections_lock:
            if ip in self.clients:
                print(f"Client {ip} already exists, updating pipe.")
            else:
                print(f"Adding new client {ip} to server.")
            self.clients[ip] = pipe
         
            
    def check_client(self, ip):
        with self.client_connections_lock:
            return ip in self.clients and self.clients[ip] is not None
        
    
    def close_clients_pipes(self):
        for ip in self.clients:
            self.clients[ip].stdin.close()
            self.clients[ip].wait()
     
     
    def close_client_pipe(self, ip):
        if self.check_client(ip):
            self.clients[ip].stdin.close()
            self.clients[ip].wait()
            print(f"Closed pipe for client {ip}.")
        else:
            print(f"No pipe found for client {ip} to close.")
         
                    
    def reboot(self):
        print("Rebooting server...")
        os.system('sudo reboot')