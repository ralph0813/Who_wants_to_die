# Who_wants_to_die?
It's a yolov3 based project.
This project fullly based on https://github.com/ultralytics/yolov3.
About how to train/infer , you can go to https://github.com/ultralytics/yolov3 for referance.

# Description
The repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows.It trained on wearing_helmet_or_not dataset.It is now private.But you can use the traind weights in the weights dictionary.
Well,I think it's meaningless to run it on the laptop or desktop.When deployed to the cloud, it well be productive.
So, there is a server.py also a client.py.
![image](https://tva1.sinaimg.cn/large/006y8mN6ly1g87goyy9quj30q40hejw7.jpg)
![image](https://tva1.sinaimg.cn/large/006y8mN6ly1g87gowcx21j30eq09pdhg.jpg)

# Requirements
Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

waitress
face_recognition
flask
requests
numpy
opencv-python
torch >= 1.2
matplotlib
pycocotools
tqdm
tb-nightly
future
Pillow



# How to use
download weights: https://pan.baidu.com/s/1QwqA3N92BgoI57Y6DOeQ0w Code：Y52j
 
`git clone https://github.com/ralph0813/Who_wants_to_die.git`

`pip3 install -U -r requirements.txt`

download weights baidu: https://pan.baidu.com/s/1QwqA3N92BgoI57Y6DOeQ0w Code：Y52j

Google Drive :https://drive.google.com/open?id=1s0eLELJNOxsuU7B_0zPmb3UhSj_HD9hq

Copy the .weights to weights forder.

`cd Who_wants_to_die`

`python3 server.py`

Another terminal:
`python3 client.py --file data/samples/timg.jpeg` or `python3 client.py --file /path/to/your/picture/`

If you wants to add other faces,find a Positive face photo and put it into `data/known_faces/` after restart server,it will be registered.
