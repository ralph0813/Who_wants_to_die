from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import os
import time
from datetime import timedelta
import sys
from flask_wtf import FlaskForm

from wtforms import SubmitField, StringField

from .face_recognice import load_known_faces, recognize, update_known_faces

sys.path.append('../')
from simple_detect import load_model, process_img

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
        else:
            return jsonify({"error": 1001, "msg": "No face found, try another photo!"})

        return render_template('upload_ok.html', text=name,
                               file_url=file_url,
                               val1=time.time())
    return render_template("upload_face.html", form=form)


@app.route('/show_face')
def show_face():
    images = []

    for face_img in os.listdir(os.path.join(basepath, "static/known_faces")):
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
        return render_template('result.html', name=result[0][0], result=result[0][1],
                               img_file=os.path.join("static/processed_images/", file_name),
                               val1=time.time())

    return render_template('detect.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
