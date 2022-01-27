#!/usr/bin/env python
# coding: utf-8

# # Image Classifier App

# In[2]:


# Imports
from fastai.vision.all import *
from fastai.vision.widgets import *
import urllib.request


# In[3]:


# Use this cell when the learner pkl file is saved in Github

# # Load model
# path = Path()
# learn_inf = load_learner(path/'export.pkl', cpu=True)


# In[4]:


# Use this cell when the learner pkl file is saved in Google Drive

# Use below link to generate downloadable link for large google drive folders:
# https://www.wonderplugin.com/online-tools/google-drive-direct-link-generator/
# NN API Key: AIzaSyAzjsmSCG4Xm4OJhnnfQRQ4iUj1VYIJZG4

# MODEL_URL = "https://www.googleapis.com/drive/v3/files/1QQb2rJpeglaCCf11TaKMzqlaYg5hWjZR?alt=media&key=AIzaSyAzjsmSCG4Xm4OJhnnfQRQ4iUj1VYIJZG4"
# urllib.request.urlretrieve(MODEL_URL, "export.pkl")

# path = Path()
# learn_inf = load_learner(path/'export.pkl', cpu=True)


# In[5]:


path = Path()
learn_inf = load_learner('mf_learner.pkl', cpu=True)


# In[6]:


btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()
confidence_pred = widgets.Label()
btn_run = widgets.Button(description='Classify')

def confidence_statement(confidence):
    if 0.8 < confidence < 0.9:
        confidence_pred.value = "I'm not too confident in this prediction..."
    elif 0.7 < confidence < 0.8:
        confidence_pred.value = "This prediction is most likely wrong..."
    elif confidence < 0.7:
        confidence_pred.value = "This prediction is most definitely wrong..."
    else:
        confidence_pred.value = " "

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(1000,1000))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]*100:.02f} %'
    confidence_statement(probs[pred_idx])
    
btn_run.on_click(on_click_classify)


# # Upload an image of either a trench, manhole or neat area and click Classify!
# 
# #### Images outside of this scope will be incorrectly classified

# In[7]:


VBox([widgets.Label('Select your image!'), 
      btn_upload, btn_run, out_pl, lbl_pred, confidence_pred])

