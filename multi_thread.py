# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:21:22 2023

@author: zareii
"""

#impors :

import face_recognition
import os, sys
import cv2
import numpy as np
import math
import mediapipe as mp
#import tensorflow as tf
from tensorflow.keras.models import load_model
from time import sleep
import requests as req 
##multi_thread
from threading import Thread
import datetime
#from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
# import argparse
# import imutils

#funcs : 
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
    
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        
        if not self.stream.isOpened() :
            sys.exit('Video source not found...')
        
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class tcp_call :
    
    def __init__(self) :
        
        self.url = 'http://192.168.43.188/'
        self.c0 = 'LED=OFF'
        self.c1 = 'LED=ON'
        self.result = -1
        self.last_call = None
        self.call_sign = None
        self.stopped = False
        self.no_connection = False
        self.number_of_failed = 0
        
    def start(self):
            # start the thread to read frames from the video stream
            Thread(target=self.tcp_call, args=()).start()
            return self

    def tcp_call(self) :
        
        if self.no_connection :
            sys.exit('Connection Failed ...')
            #print('Connection Failed ...')
        ######
        while True :
            #print(self.number_of_failed)
            if self.call_sign == 'c0' and self.last_call != 'c0':
                try :
                    req.get(self.url+self.c0,timeout=0.3) 
                    self.last_call = 'c0'
                except :
                    self.number_of_failed +=1
                    
            if self.call_sign == 'c1' and self.last_call != 'c1' :
                try :
                    req.get(self.url+self.c1,timeout=0.3)
                    self.last_call = 'c1'
                except : 
                    self.number_of_failed +=1
                    
            if self.stopped :
                return 
            
            if self.number_of_failed > 10:
                self.no_connection = True
                break
                
            sleep(0.2)
            
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def call(self,call) :
        # sets the sign which will be called
        self.call_sign = call
        
        
class VisualRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        self.ini_hands_recognition()
        self.process_current_frame = True
        self.call_sign = 'none'
        self.last_call = 'none'
        self.confirmed_names = ['muhammad','ata','ali'] 
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)
        
    def ini_hands_recognition(self) :
        # initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.handmodel = load_model('mp_hand_gesture')

        # Load class names
        f = open('gesture.names', 'r')
        self.handclassNames = f.read().split('\n')
        f.close()
    
    #######################################################
    #######################################################
    #######################################################
    def run_recognition(self,conf_threshold=90):

        name = 'none'
        handgesture = 'none'
        name = 'none'
        handgesture = 'none'
        permit = 0
        #################################################         
        video_capture = WebcamVideoStream(src=0).start()
        tcp_call0 = tcp_call()
        tcp_call0.start()
            
        while(1) :
            frame = video_capture.read()
            permit = 0
            # Only process every other frame of video to save time
            if self.process_current_frame and video_capture.grabbed :
                    
                x, y, c = frame.shape
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                fliped_frame = frame
                #fliped_frame = cv2.flip(frame, 1)
                fliped_frame_rgb = fliped_frame [:, :, ::-1]
    
                # Get hand landmark prediction
                self.result = self.hands.process(fliped_frame_rgb)
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
    
                self.face_names = []
                self.ids = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '-'
    
                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        name = name.split('.')[0]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                    if confidence == '-' :
                        conf_number = 0
                    else :
                        conf_number = float(confidence.replace('%', ''))
                    self.ids.append([name,conf_number])
                    self.face_names.append(f'{name} ({confidence})')
            self.process_current_frame = not self.process_current_frame

            # Display the results

            for (top, right, bottom, left), name , i_d in zip(self.face_locations, self.face_names,self.ids):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                rl_name = i_d [0]
                #print(rl_name)
                if rl_name in self.confirmed_names :
                    conf_number = i_d[1]
                    if conf_number > conf_threshold :
                        frame_color = (0,255,0)
                    else :
                        frame_color = (0,128,255)
                else :
                    conf_number = 0
                    frame_color = (0,0,255)

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), frame_color , 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), frame_color , cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            if len(self.ids) :
                for i_d in self.ids :
                    thename = i_d[0] 
                    cofi = i_d[1]
                    if thename not in self.confirmed_names :
                        permit = 0
                        frame_color = (0,0,255)
                    elif cofi < conf_threshold :
                        permit = 0
                        frame_color = (0,128,255)
                    else :
                        permit = 1
                        frame_color = (0,255,0)
            else :
                permit = 0 
                frame_color = (0,0,255)
            if self.result.multi_hand_landmarks:
                landmarks = []
                for handslms in self.result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # print(id, lm)
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
    
                        landmarks.append([lmx, lmy])
    
                    # Drawing landmarks on frames
                    self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS)
    
                    # Predict gesture
                    prediction = self.handmodel.predict([landmarks])
                    # print(prediction)
                    classID = np.argmax(prediction)
                    className = self.handclassNames[classID]
                    handgesture = className
                    # show the prediction on the frame
                    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 
                                1,frame_color, 2, cv2.LINE_AA)
    
            # Show the final output
            cv2.imshow("Output", frame)
                
            # print('name :',name)
            # print('type :',type(name))
            # 
            if permit :
                if handgesture == 'thumbs up' :
                    tcp_call0.call('c0')
                elif handgesture == 'thumbs down' :
                    tcp_call0.call('c1')
            #print(name,handgesture)
    
            
            if cv2.waitKey(1) == ord('q'):
                break
        # Release handle to the webcam
        video_capture.stop()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    vr = VisualRecognition()
    vr.run_recognition()