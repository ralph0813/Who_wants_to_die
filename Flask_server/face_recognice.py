import face_recognition
import os
import cv2
import numpy as np


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


# Init face recognice
def load_known_faces(face_dir='static/known_faces'):
    known_face_encodings = []
    known_face_names = []
    print("Already registered\t")
    i = 0
    for file in os.listdir(face_dir):
        if file.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            image = face_recognition.load_image_file(os.path.join(face_dir, file))
            if face_recognition.face_encodings(image):
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                name = file.split('-')[0]
                known_face_names.append(name)
                print(file + ' \t' + "as " + '\t' + name + ' \t')
                i = i + 1
    print('\n')
    return known_face_encodings, known_face_names


def update_known_faces(img_pth='/Users/ralph/github/Who_wants_to_die/Flask_server/static/known_faces'):
    file = img_pth.split("/")[-1]
    if img_pth.split('.')[-1] in ['jpg', 'jpeg', 'png']:
        image = face_recognition.load_image_file(img_pth)
        face_encoding = face_recognition.face_encodings(image)[0]
        name = file.split('-')[0]
        print(file + ' \t' + "as " + '\t' + name + ' \t')
        return face_encoding, name


def recognize(image_path, res, known_face_encodings, known_face_names):
    img = cv2.imread(image_path)
    image_name = image_path.split('/')[-1]
    face_locations = get_face_boxes_dlib(res)
    results = []
    face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[int(best_match_index)]
        face_names.append(name)

    for (top, right, bottom, left, hat_or_head), name in zip(face_boxes(res), face_names):
        result = [name, hat_or_head, (left, top, right, bottom)]
        results.append(result)
    return results
