import serial
import time
import threading

from src.command import Command
from src.event_types import REQUEST_TRACKING_EVENT, STOP_TRACKING_EVENT, CHANGE_ROI_FROM_UART_EVENT
from src.event import post_event

def serial_transmit(data):
    if ser.is_open:
        ser.write(str(data).encode() + b'\n')
    else:
        print("Error: cannot send the data because serial port is closed!")
    
def serial_transmit_binary(data_binary):
    if ser.is_open:
        ser.write(data_binary)
    else:
        print("Error: cannot send the data because serial port is closed!")
  
def open_serial():
    print("Opening serial port...")
    if not ser.is_open:
        ser.open()
    else:
        print("Serial port is already opened.")        
         
def close_serial():
    close_uart_event.set()
    if ser.is_open:
        ser.close()  
        
def reopen_serial():
    ser.close()
    time.sleep(serial_reopen_timeout)
    ser.open()
    print("Serial port is reopened.")
    
def decode_data(data):
    decoded_data = data.decode("utf-8").strip()
    print("decoded uart data:", decoded_data)
      
def handle_data(data):
    if data == Command.UART_START_TRACKING:
        post_event(REQUEST_TRACKING_EVENT)
    elif data == Command.UART_STOP_TRACKING:
        post_event(STOP_TRACKING_EVENT)
    elif data == Command.UART_CHANGE_ROI:
        value_bytes = ser.readline()
        if value_bytes:
            decoded_value = int(value_bytes.decode("utf-8").strip())
            post_event(CHANGE_ROI_FROM_UART_EVENT, decoded_value)
        else:
            print("Bad ROI values", value_bytes) 
       
def serial_receive_loop():
    while not close_uart_event.is_set():
        try:
            if not ser.is_open:
                print("Serial port is closed")
                close_uart_event.set()             
            data_bytes = ser.readline() 
            if not data_bytes:
                continue
            handle_data(decode_data(data_bytes))
        except serial.SerialException as e:
            print(f"Error occured in serial receive loop: {e}\nTrying to reopen serial port...")
            reopen_serial()
        except (TypeError, UnicodeDecodeError):
            pass
        except KeyboardInterrupt:
            close_serial()
            raise

ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyS0'
ser.timeout = 0.5
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.stopbits = 1
close_uart_event = threading.Event()
serial_reopen_timeout = 2