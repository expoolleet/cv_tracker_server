from zeroconf import Zeroconf, ServiceInfo
import socket

service_type = '_http._tcp.local.'
service_name = 'Streaming._http._tcp.local.'


def register_zeroconf(ip, port, stream_protocol, stream_port, tracking_frame_size):
    zeroconf = Zeroconf()
    data = {
               'server_ip': ip,
               'stream_protocol' : stream_protocol,
               'server_port': port,
               'stream_port': stream_port,
               'tracking_frame_size': tracking_frame_size
          }
    print(f"Zeroconf data: {data}")
    info = ServiceInfo(service_type,
                   service_name,
                   port=port,
                   properties=data,
                   server=socket.gethostname()+'.local.')
    zeroconf.register_service(info)
    print(f"Zeroconf service registered with sevice type '{service_type}' with name '{service_name}'.\n")
    return zeroconf