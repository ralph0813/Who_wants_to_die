import flask
import sys
sys.path.append('../')
from models import *
from utils.utils import *
import face_recognition
import cv2
import numpy as np
# from locate_face import get_face_boxes_dlib, load_model
import pyhdfs
from waitress import serve


from simple_detect import load_model, process_img
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import os
import time
from datetime import timedelta
import sys
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from face_recognice import recognize, update_known_faces
from flask_uploads import UploadSet, configure_uploads, patch_request_class
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired


basepath = os.path.dirname(__file__)  # 当前文件所在路径
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join(basepath, 'static/')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_FACES_DEST'] = os.path.join(UPLOAD_FOLDER, "known_faces")

images = UploadSet('faces', ALLOWED_EXTENSIONS)
configure_uploads(app, images)
patch_request_class(app)

fs = pyhdfs.HdfsClient(hosts='localhost:50070', user_name='root')

def formate_image(img, new_shape=416, color=(128, 128, 128)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
    dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratiow, ratioh, dw, dh


def init_network(cfg, data, weights, img_size=416):
    # Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True  # set False for reproducible results
    # Initialize model
    model = Darknet(cfg, img_size)
    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    # Eval mode
    model.to(device).eval()
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    init = (classes, colors, model, img_size, device)
    return init


def get_bbox(init, image,image_name,conf_thres=0.5, nms_thres=0.5):
    classes, colors, model, img_size, device = init
    file_np_array = np.frombuffer(image, np.uint8)
    img0 = cv2.imdecode(file_np_array, cv2.IMREAD_COLOR)
    img, *_ = formate_image(img0, new_shape=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
    img /= 255.0

    res = []
    # Get detections
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred, _ = model(img)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls_conf, cls in det:
            # Add bbox to the image

            label = '%s %.2f' % (classes[int(cls)], conf)
            left, top, right, bottom = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            classname = label.split(" ")[0]
            result = [classname, left, top, right, bottom]
            res.append(result)

    if len(res) == 0:
        cv2.imwrite(os.path.join('output',image_name), img0)
        return
    else:
        face_locations = []
        face_encodings = []
        face_names = []
        face_locations = get_face_boxes_dlib(res)
        results = []

        face_encodings = face_recognition.face_encodings(img0, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left, hat_or_head), name in zip(face_boxes(res), face_names):
            # Draw a box around the face
            cv2.rectangle(img0, (left, top), (right, bottom+20), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img0, (left, bottom - 15), (right, bottom+20), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(img0, name, (left + 6, bottom), font, 0.7, (0, 255, 0), 1)
            cv2.putText(img0, hat_or_head, (left + 6, bottom + 20), font, 0.7, (255, 0, 0), 1)
            cv2.imwrite(os.path.join('output',image_name), img0)
    #         result = [name,hat_or_head,(left,top,right,bottom)]
            result = [name,hat_or_head]
            results.append(result)
        return results


def singal_detcet(init, image_name):
    ori_image_path = os.path.join('/root/original_img/',image_name)
#     print(ori_image_path)
    image = fs.open(ori_image_path).read()
    res = get_bbox(init, image, image_name)
    fs.copy_from_local(os.path.join('/home/ralph/project/Who_wants_to_die/Flask_server/output',image_name),
                       os.path.join('/root/handled_img',image_name))
    print('fs.copy success:',os.path.join('/home/ralph/project/Who_wants_to_die/Flask_server/output',image_name),
          "to",os.path.join('/root/handled_img',image_name))
    return res

def get_face_boxes_dlib(res):
    face_boxes_dlib=[]
    for face in range(len(res)):
        face_location = res[face][1:]
        xmin = face_location[0]
        ymin = face_location[1]
        xmax = face_location[2]
        ymax = face_location[3]
        ymin = ymax * 0.9 - (ymax - ymin) * 0.5
        ymax = ymax - (ymax - ymin) * 0.1
        face_box_dlib = (int(ymin), xmax, int(ymax), xmin)
        face_boxes_dlib.append(face_box_dlib)

    return face_boxes_dlib

def face_boxes(res):
    face_boxes=[]
    for face in range(len(res)):
        hat_or_head =  res[face][0]
        face_location = res[face][1:]
        xmin = face_location[0]
        ymin = face_location[1]
        xmax = face_location[2]
        ymax = face_location[3]
        face_box = (ymin, xmax, ymax, xmin, hat_or_head)
        face_boxes.append(face_box)
    
    return face_boxes



#### 初始化安全帽检测
def load_model():
    out_path = os.path.join('output')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    cfg = '/home/ralph/project/yolov3/cfg/hat_608.cfg'
    data = '/home/ralph/project/yolov3/data/hat_608.data'
    weights = '/home/ralph/project/yolov3/weights/hat_608.weights'
    with torch.no_grad():
        # init
        init = init_network(cfg, data, weights)
    
    
    return init
    
    

### 初始化人脸比对    
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    print("已注册人脸表\t")
    i = 0
    basedir = '/home/ralph/project/Who_wants_to_die/Flask_server/static/known_faces'
    for file in os.listdir(basedir):
        if file.split('.')[-1] in ['jpg','jpeg','png']:
            image = face_recognition.load_image_file(os.path.join(basedir,file))
            """
            换用yolo
            """
            # print(face_recognition.api.face_locations(image, number_of_times_to_upsample=1, model='hog')[0])
            # im = cv2.imread(line)
            # face_bounding_boxes = get_face_boxes_dlib(im)
            # print(face_bounding_boxes)
            # print("*" * 20)
            # face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
            res = face_recognition.face_encodings(image)
            if res :
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(file.split('-')[0])
                print(file + " \t" + file.split('.')[0])
                i = i + 1
    
    return known_face_encodings,known_face_names


class UploadForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired("Name can not be empty.")])
    image = FileField('选择上传的文件', validators=[
        DataRequired(),
        FileAllowed(images, '只能上传图片！'),
        FileRequired('文件未选择！')])

    submit = SubmitField('上传')


# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=5)

# 加载模型
known_face_encodings, known_face_names = load_known_faces()
model = load_model()


def allowed_file(filename):
    if filename.lower().rsplit('.')[-1] in ALLOWED_EXTENSIONS:
        return filename


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload_face', methods=['POST', 'GET'])
def upload_face():
    global known_face_encodings, known_face_names
    form = UploadForm()
    if form.validate_on_submit():
        name = form.name.data
        filename = form.image.data.filename
        filename = name + "-" + str(int(time.time())) + "." + filename.split(".")[-1]
        images.save(form.image.data, name=filename)
        file_url = images.url(filename)
        # update Known_Faces
        res = update_known_faces(os.path.join(basepath, 'static/known_faces/', filename))
        if res:
            known_face_encoding, known_face_name = res
            known_face_encodings.append(known_face_encoding)
            known_face_names.append(known_face_name)
            return render_template('upload_ok.html', text=name, file_url=file_url, val1=time.time())
        else:
            return jsonify({"error": 1001, "msg": "No face found, try another photo!"})
        
        
    return render_template("upload_face.html", form=form)


@app.route('/show_face')
def show_face():
    images = []

    for face_img in os.listdir(os.path.join(basepath, "static/known_faces")):
        if face_img.split('.')[-1] in ALLOWED_EXTENSIONS:
            name = face_img.split('-')[0]
            img_path = os.path.join("static/known_faces", face_img)
            image = [name, img_path]
            images.append(image)
    return render_template("show_face.html", images=images)


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Only png、jpg、jpeg are allowed!"})

        file_name = str(int(time.time())) + f.filename
        upload_img_path = os.path.join(basepath, 'static/origin_images',
                                       file_name)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_img_path)
        processed_img_path = os.path.join(basepath, 'static/processed_images')
        res = process_img(ori_img_path=upload_img_path, out_img_name=file_name, img_out_dir=processed_img_path)
        result = recognize(upload_img_path, res, known_face_encodings, known_face_names)
        if result:
            name=result[0][0]
            if result[0][1] == "no_helmet":
                information = "This guy does not wear a helmet, he may die!"
            else:
                information = "This guy wears a helmet, he is a good guy!"
        else :
            name = "Unknow"
            information = "Error"
            
        return render_template('result.html', name=name, result=information, 
                               img_file=os.path.join("static/processed_images/", file_name),
                               val1=time.time())

    return render_template('detect.html')









@app.route("/hat_predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    if flask.request.method == 'POST':
        if flask.request.get_json():
            image = flask.request.json["image"]
            image_name = image
            r = singal_detcet(model, image_name)
            data['predictions'] = r
            data["success"] = True
            res = flask.jsonify(data)
            print(image,data)
    # Return the data dictionary as a JSON response.
    return res 

    


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    serve(app, host='0.0.0.0', port=8000)
    
    
    
    
    
    
