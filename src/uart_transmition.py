import serial
import time
import RPi.GPIO as GPIO

from src.command import Command
from src.event_types import REQUEST_TRACKING, STOP_TRACKING, CHANGE_ROI_FROM_UART
from src.event import post_event

#№GPIO.setmode(GPIO.BCM)
#GPIO.setup(15, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

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
    global is_uart_closed
    is_uart_closed = True
    if ser.is_open:
        ser.close()
            
def serial_receive_loop():
    while not is_uart_closed:
        try:
            data_bytes = ser.readline() 
            if not data_bytes:
                #post_event(STOP_TRACKING, True)
                continue
            #else:
            print("data bytes:", data_bytes)
            decoded_data = data_bytes.decode("utf-8").strip()
            print("decoded data:", decoded_data)
            if decoded_data == Command.UART_START_TRACKING:
                post_event(REQUEST_TRACKING)
            elif decoded_data == Command.UART_STOP_TRACKING:
                post_event(STOP_TRACKING)
            elif decoded_data == Command.UART_CHANGE_ROI:
                value_bytes = ser.readline()
                decoded_value = int(value_bytes.decode("utf-8").strip())
                post_event(CHANGE_ROI_FROM_UART, decoded_value)
                #print("Decoded int value:", decoded_value)
        except serial.SerialException as e:
            if not is_uart_closed:
                print(f"Error occured in serial receive loop: {e}\nTrying to reopen serial port...")
                ser.close()
                time.sleep(2)
                ser.open()
                print("Serial port is reopened.")    
        except UnicodeDecodeError:
            pass
            #print(f"Warning: Error when decoding data with utf-8 sent by serial: {data_bytes!r}")
        except Exception as e:
            print(f"Warning: Error occurred when decoding data sent by serial: {e}")
            ser.close()


ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyS0'
ser.timeout = 0.5
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.stopbits = 1
is_uart_closed = False
# if not ser.is_open:
#     ser.open()
# print("Serial port is opened.")


