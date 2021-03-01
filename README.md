# Facial-Expression-Controller
Use your face as a video game controller

## Background

This project was created for my master thesis at BTH.

The thesis can be found here:
http://urn.kb.se/resolve?urn=urn:nbn:se:bth-12924

## How it works

The TobiInterface applicaiton is using a facial recognition network called CLM. CLM can detect various key points on a face. My application will calculate the distances between the points in order to determine when a certain facial feature is moving. 

Facial features has been tied to joystick controlls. For example, the eyebrows can be tied to a jump button.

When a facial feature is detected to have been moved, a joystick key press is sent to the vJoystick application. Thus the facial expression can be used as a game controller 


The TobiPlatformer is a simple platformer game used for testing and evaluation. To start the game, you need to enter a number, and press the start button. The applicaiton may take some time to load.


## Demo Video
This video shows the TobiInterface in acction, reading the facial expressions of the player and interpretating them as a joystick controller.

[![Demo Video](https://i.ytimg.com/vi/sSWgdPzO4mM/hqdefault.jpg)](https://youtu.be/sSWgdPzO4mM)
