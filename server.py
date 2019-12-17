import flask
from models import *
from utils.utils import *
import face_recognition
import cv2
import numpy as np
from waitress import serve
import time


def formate_image(img, new_shape=416, color=(128, 128, 128)):
    shape = img.shape[:2]
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


def handel_image(init, image, image_name, conf_thres=0.5, nms_thres=0.5):
    classes, colors, model, img_size, device = init
    img0 = image
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

    face_locations = get_face_boxes_dlib(res)
    results = []
    face_encodings = face_recognition.face_encodings(img0, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[int(best_match_index)]
        face_names.append(name)

    for (top, right, bottom, left, hat_or_head), name in zip(face_boxes(res), face_names):
        # Draw a box around the face
        cv2.rectangle(img0, (left, top), (right, bottom + 20), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(img0, (left, bottom - 15), (right, bottom + 20), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img0, name, (left + 6, bottom), font, 0.7, (0, 255, 0), 1)
        cv2.putText(img0, hat_or_head, (left + 6, bottom + 20), font, 0.7, (255, 0, 0), 1)
        cv2.imwrite(os.path.join('output', image_name), img0)
        result = [name, hat_or_head, (left, top, right, bottom)]
        results.append(result)

    return results


def singal_detcet(init, image):
    image_name = "out_" + str(int(time.time())) + ".jpg"
    file_np_array = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(file_np_array, cv2.IMREAD_COLOR)
    res = handel_image(init, image, image_name)
    print(res)
    return res


def get_face_boxes_dlib(res):
    face_boxes_dlib = []
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
    face_boxes = []
    for face in range(len(res)):
        hat_or_head = res[face][0]
        face_location = res[face][1:]
        xmin = face_location[0]
        ymin = face_location[1]
        xmax = face_location[2]
        ymax = face_location[3]
        face_box = (ymin, xmax, ymax, xmin, hat_or_head)
        face_boxes.append(face_box)

    return face_boxes


app = flask.Flask(__name__)


# loading weights
def load_model():
    out_path = os.path.join('output')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cfg = 'cfg/hat_608.cfg'
    data = 'data/hat_608.data'
    weights = 'weights/hat_608.weights'
    with torch.no_grad():
        # init
        init = init_network(cfg, data, weights)

    return init


# Init face recognice
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    print("Already registered\t")
    i = 0
    basedir = 'data/known_face'
    for file in os.listdir(basedir):
        if file.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            image = face_recognition.load_image_file(os.path.join(basedir, file))
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            name = file.split('.')[0]
            known_face_names.append(name)
            print(file + ' \t' + "as " + '\t' + name + ' \t')
            i = i + 1
    print('\n')
    return known_face_encodings, known_face_names


@app.route("/hat_predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image_name = image
            results = singal_detcet(helmet_init, image_name)
            print(results)
            data['predictions'] = results
            data["success"] = True
        else:
            data['predictions'] = "NULL"
            data["success"] = False
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    helmet_init = load_model()
    known_face_encodings, known_face_names = load_known_faces()
    serve(app, host='0.0.0.0', port=5000)

