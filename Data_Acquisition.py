 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:11:04 2021

@Editor: GESTAR Project Team
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:49:18 2021

@Editor: GESTAR Project Team
"""

# !/usr/bin/python3
#######################################
# Copyright (c) Acconeer AB, 2020-2021
# Edited by: GESTAR Project Team
# All rights reserved
#######################################
"""
This file is originally provided by Acconeer and has been fine tuned and edited by GESTAR Project Team for data acquisition purpose. 
"""
import argparse
import sys
import time
import serial
import struct
import math
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import keyboard
Sweeps = 18
sps = 207
lo_lim = 0
up_lim = 207
sweeps=80
samples=620
duration = 0
# import keras
# import tensorflow
#import seaborn as sns


class ModuleError(Exception):
    """
    One of the error bits was set in the module
    """
    pass


class ModuleCommunication(object):
    """
    Simple class to communicate with the module software
    """

    def __init__(self, port, rtscts):
        self._port = serial.Serial(port, 115200, rtscts=rtscts,
                                   exclusive=True, timeout=2)
    def change_baud(self, baud):
        self._port.baudrate = 3000000

    def read_packet_type(self, type):
        """
        Read any packet of type. Any packages received with
        another type is discarded.
        """
        while True:
            header, payload = self.read_packet()
            if header[3] == type:
                break
        return header, payload

    def read_packet(self):
        header = self._port.read(4)
        length = int.from_bytes(header[1:3], byteorder='little')

        data = self._port.read(length + 1)
        assert data[-1] == 0xCD
        payload = data[:-1]
        return header, payload

    def register_write(self, addr, value):
        """
        Write a register
        """
        data = bytearray()
        data.extend(b'\xcc\x05\x00\xf9')
        data.append(addr)
        data.extend(value.to_bytes(4, byteorder='little', signed=False))
        data.append(0xcd)
        self._port.write(data)
        header, payload = self.read_packet_type(0xF5)
        assert payload[0] == addr

    def register_read(self, addr):
        """
        Read a register
        """
        data = bytearray()
        data.extend(b'\xcc\x01\x00\xf8')
        data.append(addr)
        data.append(0xcd)
        self._port.write(data)
        header, payload = self.read_packet_type(0xF6)
        assert payload[0] == addr
        return int.from_bytes(payload[1:5], byteorder='little', signed=False)

    def buffer_read(self, offset):
        """
        Read the buffer
        """
        data = bytearray()
        data.extend(b'\xcc\x03\x00\xfa\xe8')
        data.extend(offset.to_bytes(2, byteorder='little', signed=False))
        data.append(0xcd)
        self._port.write(data)

        header, payload = self.read_packet_type(0xF7)
        assert payload[0] == 0xE8
        return payload[1:]

    def read_stream(self):
        header, payload = self.read_packet_type(0xFE)
        return payload

    @staticmethod
    def _check_error(status):
        ERROR_MASK = 0xFFFF0000
        if status & ERROR_MASK != 0:
            ModuleError(f"Error in module, status: 0x{status:08X}")

    @staticmethod
    def _check_timeout(start, max_time):
        if (time.monotonic() - start) > max_time:
            raise TimeoutError()

    def _wait_status_set(self, wanted_bits, max_time):
        """
        Wait for wanted_bits bits to be set in status register
        """
        start = time.monotonic()

        while True:
            status = self.register_read(0x6)
            self._check_timeout(start, max_time)
            self._check_error(status)

            if status & wanted_bits == wanted_bits:
                return
            time.sleep(0.1)

    def wait_start(self):
        """
        Poll status register until created and activated
        """
        ACTIVATED_AND_CREATED = 0x3
        self._wait_status_set(ACTIVATED_AND_CREATED, 3)

    def wait_for_data(self, max_time):
        """
        Poll status register until data is ready
        """
        DATA_READY = 0x00000100
        self._wait_status_set(DATA_READY, max_time)




counter = 0
iq_data = []
real = []
matrix = []
sweep_mat = np.zeros(3726)
sweep_mat_dup = []
distance = np.linspace(200, 600, 207)


counter=0







def streaming_mode():
    duration = 1
    data=[]
    start = time.monotonic()
    global iq_data
    count=0
    file_name=time.strftime("%Y%m%d%H%M%S.npy")
    #path = 'E:/NWN/gunshot/model_training/Training data/EngineIdling_urban8'
    #f = open('E:\\SPCAI\\Dataset\\Swipeleft\\'+file_name, "a")
    counter=0
    start = time.monotonic()
    loopcount=0
    #while loopcount<=80:
    while loopcount<        80:
        
        loopcount=loopcount+1
        
        counter=counter+1
        stream = com.read_stream()
        assert stream[0] == 0xFD
        result_info_length = int.from_bytes(stream[1:3], byteorder='little')
        if(count==3726):
            count=0

        buffer = stream[26:]
        a = struct.unpack("<ff", stream[32:40])

        for i in range(0, len(buffer), 2):
            a1 = struct.unpack('<e', buffer[i:i + 2])
            for a in a1:
                iq_data.append(a)
                # matrix.append(a)
                #sweep_mat[count]=a
                data.append(a)
                #f.write(str(a))
                #f.write('\n')
                #np.save('E:\\SPCAI\\Dataset\\Swipeleft\\'+file_name, a)

            count=count+1
    return data


    #f.close()


def module_software_test(port, flowcontrol, streaming, duration):
    """
    A simple example demonstrating how to use the distance detector
    """
    print(f'Communicating with module software on port {port}')

    # duration=1
    global com
    com = ModuleCommunication(port, flowcontrol)


    # Make sure that module is stopped
    com.register_write(0x03, 0)

    # Give some time to stop (status register could be polled too)
    time.sleep(0.5)

    # Clear any errors and status
    com.register_write(0x3, 4)

    # Read product ID
    product_identification = com.register_read(0x10)
    print(f'product_identification=0x{product_identification:08X}')

    version = com.buffer_read(0)
    print(f'Software version: {version}')

    if streaming:

        com.register_write(5, 1)
        com.register_write(0x2, 0x02)
        com.register_write(0x23, 100000)
        com.register_write(0x21, 400)
        com.register_write(0x20, 200)
        com.register_write(0x29, 2)
        global baudrate
        baudrate = 3000000
        print('step 2')

        com.register_write(0x07, baudrate)
        com.change_baud(baudrate)
        print('step 3') 
        com.register_write(0x25, 3)
        com.register_write(0x28, 2)
        com.register_write(0x22, 2)

        com.register_write(0x20, 400)
        com.register_write(0x21, 600)
        com.register_write(0x24, 500)
        com.register_write(0x29, 2)
        com.register_write(0x30, 5)
        com.register_write(0x40, 0)
        com.register_write(0x31, 1)
        com.register_write(0x32, 0)
        com.register_write(0x26, 0)
        com.register_write(0x33, 1)
        com.register_write(0x25, 3)
        com.register_write(0x34, 6)
        com.register_write(0x05, 1)
        print('step length', com.register_read(0x85))


    else:
        # Set mode to 'distance'
        com.register_write(0x2, 0x200)

    # Activate and start
    #          com.register_write(3, 3)

    if streaming:
        while True:
            # if keyboard.read_key() == "space":

             print("Button pressed")
             #time.sleep(1)
             import winsound
             frequency = 2500  # Set Frequency To 2500 Hertz
             duration = 500     # Set Duration To 1000 ms == 1 second
             winsound.Beep(frequency, duration)
             time.sleep(0.7)
             winsound.Beep(frequency, 1200)
            #    time.sleep(0.5)
             com.register_write(3, 3)
             #time.sleep(0.129) 
             print("Activated")
             
             f=np.load('preprocessing/avg_noise/avg_noise.npy')
             # print(f)
             d=  streaming_mode()
             d=d-f
             print('data length',len(d))
             com.register_write(0x03, 0)
             print('Closed')
             winsound.Beep(frequency, duration) 
     
     
             #gesture_prediction (d) 
             #print(d )
             #d=np.array(d)
             #print(d.shape)
             file_name = time.strftime("%Y%m%d%H%M%S.npy")
     
            # np.save('data80x622/no_gesture/' + file_name,d)
             print('Saved')
     
             #plot_3M(d)
             heatmaps (d)




  
        

    print()                                                                         
    print('End of example')
    com.register_write(0x03, 0)
    #plot_sweep(iq_data)


def heatmaps(gesture):
    gesture=np.array(gesture)
   
    gesture = gesture.reshape((80, 620))
    gesture=np.swapaxes(gesture,0,1)
    

    plt.imshow(gesture, interpolation='nearest')
  
    #plt.show()
    plt.axis('off')
    plt.pause(1)
    plt.close()


def main():
    """
    Main entry function
    """
    parser = argparse.ArgumentParser(description='Test UART communication')
    parser.add_argument('--port', default="COM4",
                        help='Port to use, e.g. COM1 or /dev/ttyUSB0')
    parser.add_argument('--no-rtscts', action='store_true',
                        help='XM132 and XM122 use rtscts, XM112 does not')
    parser.add_argument('--duration', default=1,
                        help='Duration of the example', type=int)
    parser.add_argument('--streaming', action='store_true',
                        help='Use UART streaming protocol')
    # plt.rcParams['animation.html']='jshtml'
    args = parser.parse_args()
    global duration
    module_software_test(args.port, args.no_rtscts, not args.streaming, args.duration)

  


if __name__ == "__main__":

    sys.exit(main())




