import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import PIL
import pandas as pd
import numpy as np
import cv2


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        return opencv_image
    else:
        return None

def load_model():
    model = YOLO('best.pt')
    return model


def load_labels():
    labels_path = 'classes.txt'
    labels_file = os.path.basename(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict():
    
    model = load_model()
    img = load_image()
    data={}
    categories = load_labels()
    resulst = model.predict(img,conf=0.5,device='cpu')
    a = resulst[0].boxes.boxes.cpu()
    px = pd.DataFrame(a).astype("float")
    result_labels = []
    result_accuracies = []
    for index,row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
        t = int(px[5][index])
        r = categories[t]
        cv2.putText(img,str(r),(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        result_labels.append(r)
        result_accuracies.append(int(row[4]*100))
    data = pd.DataFrame({'Labels':result_labels,'Accuracy %':result_accuracies)})
    if len(data)!=0:
        st.image(img, channels="BGR")
        st.table(data)
    else:
        st.text("Nothing Detected")


def main():
    st.title('Object Detection Demo')
    predict()

if __name__ == '__main__':
    main()
