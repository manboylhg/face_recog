# coding= utf-8

import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

known_image_kaka = face_recognition.load_image_file('image/kaka.png')
known_image_ronaldo = face_recognition.load_image_file('image/ronaldo.jpg')

kaka_face_encoding = face_recognition.face_encodings(known_image_kaka)[0]
ronaldo_face_encoding = face_recognition.face_encodings(known_image_ronaldo)[0]

know_encodings = [
    kaka_face_encoding,
    ronaldo_face_encoding
]

image_to_test = face_recognition.load_image_file('image/kaka2.jpg')
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

face_distances = face_recognition.face_distance(know_encodings,image_to_test_encoding)

for i,face_distance in enumerate(face_distances):
    print('the test image has a distance of {:.2} from known image #{}'.format(face_distance,i))
    print('with a normal cutoff of 0.6,can the image match the known image? {}'.format(face_distance<0.6))
    # print('with a normal cutoff of 0.5,can the image match the known image? {}'.format(face_distance<0.5))