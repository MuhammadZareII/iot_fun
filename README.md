# iot_fun
this project is composed of 2 prats :
1 - python scripts : which does the bulk of the job , it can detect face and recognize certain hand gestures .
2 - ESP board programmed as server , waiting for requests from python script .
the main idea is that it (python script) sees familiar people making certain hand gestures , it sends a tcp request to an esp board , this can be used for example to turn an led on and off .
