# iot_fun
Poject discription :
uses a camera to recognize familier faces and certain hand gestures , a command is associated with each hand gesture .
the command is issued if : it recognizes the face owner name to be in the confirmed_names list (1) and the hand gesture linked with that command .

this project is has 2 prats :

1 - The python script (multi_thread.py) : it uses face_recognition package to recognize faces , it use .png files of faces in /face directory for recognition , example :it would lable a face which matches face in joe.png , joe .
and there is a part which can detect hand gestures using mp_hand_gesture model (a keras model) .
the whole thing is into threads which run concurently .

**note** : if you want the code to run properly you should modify following parts :

i- .png files in /faces directory (palce .png files of faces you want the code to recognize)

ii - self.url value in tcp_call class (multi_thread.py line 105) , palce the adress of your esp board (or whatever) in the network .

iii - names in the self.confirmed_names list (multi_thread.py line 173) the code will response to those faces whos names are in this list only



2 - ESP board programmed as server , waiting for requests from python script .
the main idea is that it (python script) sees familiar people making certain hand gestures , it sends a request to an esp board , this can be used for example to turn an led on and off .


**it's highly recommended to creat a virtual env for runnig the code**
