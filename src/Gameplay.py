# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:45:52 2021

@author: shane
"""

import pandas as pd
import numpy as np
import math
import time
from collections import namedtuple
from copy import deepcopy  
import serial

try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()
except:
    pass

global complete
complete = True

#Testing board
gameboard = [[0, 0, 1, 0, 0, 0],
             [0, 1, 1, 1, 1, 0],
             [0, 1, 2, 2, 2, 1],
             [1, 2 ,2 ,2 ,1, 0],
             [0, 1, 2 ,2 ,1, 0],
             [0, 1, 1, 1, 0, 0]]

print("original board: \n", np.asarray(gameboard))
def enclosed(data):
    #pad the board with 0 which is unoccupied
    width = len(data[0]) + 2
    height = len(data) + 2
    board = deepcopy(data)
    for y in board:
        y.insert(0,0)
        y.append(0)
    board.insert(0, [0 for x in range(width)])
    board.append([0 for x in range(width)])
    #print("padded board: \n", np.asarray(board))
    loop_sequence = 2
    remove_white = False
    remove_black = False
    #Check black first to see if there is an enclosure then check white.
    while(loop_sequence > 0 and not remove_black and not remove_white ):
        queue = [(0,0)]
        visited = [(0,0)]
        while queue:
            y, x = queue.pop(0)
            adj = ((y+1, x), (y-1, x), (y, x+1), (y, x-1))
            for n in adj:
                if (-1 < n[0] < height and -1 < n[1] < width and not n in visited and (board[n[0]][n[1]] == 0 or board[n[0]][n[1]] == 2)):
                    queue.append(n)
                    visited.append(n)
    # remove the boundaries and make decisions
        freecoords = [(y-1, x-1) for (y, x) in visited if
                     (0 < y < height-1 and 0 < x < width-1)]
        allcoords = [(y, x) for y in range(height-2) for x in range(width-2)]
        complement = [i for i in allcoords if not i in freecoords]
        
        #check black first
        if(loop_sequence ==2):
            bordercoords = [(y, x) for (y, x) in complement if gameboard[y][x] == 1]
            complement_exclude_border = [j for j in  complement if not j in bordercoords ]
            closedcoords = [(y, x) for (y, x) in complement if gameboard[y][x] == 2]
            if(len(complement_exclude_border) == len(closedcoords)):
                remove_white = True
            else:
                closedcoords = 0
        elif(loop_sequence ==1):
            bordercoords = [(y, x) for (y, x) in complement if gameboard[y][x] == 2]
            complement_exclude_border = [j for j in  complement if not j in bordercoords ]
            closedcoords = [(y, x) for (y, x) in complement if gameboard[y][x] == 1]
            if(len(complement_exclude_border) == len(closedcoords)):
                remove_black = True
            else:
                closedcoords = 0
        loop_sequence = loop_sequence - 1
        
    PixelGroups = namedtuple('PixelGroups', ['free', 'closed', 'border'])
    return PixelGroups(freecoords, closedcoords, bordercoords)

##########################################################################
#pending: edge cases
##########################################################################

def print_groups(ysize, xsize, pixelgroups):
    ys= []
    for y in range(ysize):
        xs = []
        for x in range(xsize):
            if (y, x) in pixelgroups.free:
                xs.append('.')
            elif (y, x) in pixelgroups.closed:
                xs.append('X')
            elif (y, x) in pixelgroups.border:
                xs.append('#')
        ys.append(xs)
    print('\n'.join([' '.join(k) for k in ys]))
    
#Determine which color forms the enclosure
#Find all the locations of the pieces that need to be dropped
def calculate():
    remove = enclosed(gameboard)
    print("new board: \n", np.asarray(gameboard))
    print_groups(6, 6, remove)
    print("Remove These Pieces: " + str(remove.closed))
    return np.asarray(remove.closed)

# pause the program until the action is complete
def wait():
    loop = True
    while loop:
        if ser.in_waiting > 0:
            handshake = ser.readline().decode('utf-8').rstrip()
            if(handshake):
                loop = False
                print("action complete")

#send the command via serial
def send(action, color, location):
    if complete:
        location_sum = ""
        try:
            flat = location.flatten()
            for enum in range(0,int(len(flat)/2)):
                location_con = str(flat[enum]) + str(flat[enum+1])
                location_sum = location_sum + location_con
            command = int(str(action)+str(color)+str(location_sum))
            #uncomment while running it on the actual system
            """
            ser.write(bytes(command, 'utf-8'))
            time.sleep(0.5)
            complete = False

            """
        except:
            print("Error!")
    else: 
        print("action not complete")
#first digit: action, second digit: color, after that, a pair of two digits denoting the location

#running the game, this will be the main loop
if __name__ == '__main__':
    remove = calculate()    #calculate which 
    send(1,1,remove)
    wait()

"""
    while True:
       
            code = ser.readline().decode('utf-8').rstrip()
            if code == 1: # if the arduino is ready 
                remove = enclosed(gameboard)
"""

