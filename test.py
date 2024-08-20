#!/usr/bin/env python3
from ultralytics import YOLO
from PIL import Image

# def img_callback(data):
#     global array, class_pub, det_image_pub
#     array = rosnp.numpify(data)
#     det_result = model(array)
#     det_annotated = det_result[0].plot(show=False)
#     det_image_pub.publish(rosnp.msgify(Image, det_annotated, encoding="utf-8"))

#     classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
#     names = [det_result[0].names[i] for i in classes]

#     class_pub.publish(String(data=str(names)))

# MODEL =
model = YOLO("yolov8_obb_finetuned.pt")


result = model(Image.open("image.jpg"))
