import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
# Emergency vehicle Identification
""")
file = st.file_uploader("Please upload an vehicle image", type =["jpg","png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    size =(180,180)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text("Please upload an image again")
else:
    image = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image,model)
    class_names = ['Emergency','Non-Emergency']
    string = "This image most likely is "+class_names[np.argmax(predictions)]
    st.success(string)
