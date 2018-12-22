# coding= utf-8

import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
image_dir = 'image/'
image_name = 'lhg.jpg'
image_dir_name = image_dir+image_name

image = face_recognition.load_image_file(image_dir_name)  #加载图片
face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=0,model='cnn')

for face_location in face_locations:
    top, rigth, bottom, left = face_location
    face_image = image[top:bottom,left:rigth]
    pil_image = Image.fromarray(face_image)
    pil_image.save(image_dir+'face_'+image_name)
    pil_image.show()

print(face_locations)
# plt.imshow(face_locations)
# plt.show()

#
#   ## face line
# image1 = face_recognition.load_image_file(image_dir+'face_'+image_name)
# face_landmarks_list = face_recognition.face_landmarks(image1)
# pil_image2 = Image.fromarray(image1)
# d = ImageDraw.Draw(pil_image2)
#
# for face_landmarks in face_landmarks_list:
#     for facial_feature in face_landmarks.keys():
#         d.line(face_landmarks[facial_feature],width=3)
#
# pil_image2.show()
#
#
#
#

# print(face_landmarks_list)




