#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import swifter


# In[4]:


DATASET_PATH="datasets/"
DATASET_CSV=DATASET_PATH+"styles.csv"
img_width=80
img_height=60
channels=3

def plot_images(images, nrows=1, ncols=1, figsize=(8,8)):
    fig,axes =plt.subplots(nrows=nrows, ncols=ncols, figsize= figsize)
    axes=axes.flatten()
    for img_name, ax in zip(images,axes):
        ax.imshow(cv2.cvtColor(images[img_name], cv2.COLOR_BGR2RGB))
        ax.set_title(img_name)
        ax.set_axis_off()
        
        
def img_path(PATH,img):
    return PATH+"/images/"+img

def load_image(PATH,img):
    return cv2.imread(PATH+"/images/"+img)

# Output from Neural Network is a vector (2048 nodes output embeded into 1 vector). 
# We can use this vector to find distance using cosine or jaccard formula for similarity
# We pass the model and img_name


# Get embeding (vectors) for the entire dataset
def generate_embedings_entire_dataset(model, df_images, name):
    map_embedings = df_images['image_name'].swifter.apply(lambda img: get_embeding(model, img))
    df_embd = map_embedings.apply(pd.Series)
    df_embd.to_csv(name)

    #Combining with the dataset to save it and regain easily
    df_embd_with_data=df_images.join(df_embd)
    df_embd_with_data.to_csv("Dataset_With_{}".format(name))
    
    return df_embd, df_embd_with_data



# This function is for file/image that will be uploaded
def get_embeding_uploaded(model, img):
    try:
        plt.imshow(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x).reshape(-1)
    except:
        print("Error in Image File!")
        return [0]*2048

# This function is for the file/image from the dataset
def get_embeding(model, img_name):
    try:
        img = image.load_img(img_path(DATASET_PATH,img_name), target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x).reshape(-1)
    except:
        print("Error in Image File")
        return [0]*2048
    
    
# Find the TOP N images that match to the current image using cosine distance matrix
def similarity_dataset(cosine_matrix, indices,idx, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_matrix[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    return indices.iloc[idx_rec].index, idx_sim


# Uploading image for new recommendation (OUT OF DATASET)
def get_uploaded_image(upload):
    if(len(upload.data)==1):
        nparr = np.frombuffer(upload.data[-1], np.uint8)
        img_np = cv2.imdecode(nparr, flags=1)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np2=cv2.resize(img_np,(img_height, img_width))
    else:
        print("Image NOT UPLOADED! USING DEFAULT IMAGE")
        img_np=cv2.cvtColor(cv2.imread("./test/img1.jpg"), cv2.COLOR_BGR2RGB)
        img_np2=cv2.resize(img_np,(img_height, img_width))
        
    return img_np2

def calculate_cosine_distances(df_ref, metric="cosine"):
    cosine_distance = 1-pairwise_distances(df_ref, metric=metric)
    indices = pd.Series(range(len(df_ref)), index=df_ref.index)
    
    return cosine_distance, indices


# Run this for Recommendation for UPLOADED_IMAGE
def get_recommendation_from_uploaded_image(model,upload, df_embeded, df_images):
    uploaded_image_temp=get_uploaded_image(upload)
    plt.imshow(uploaded_image_temp)
    plt.title("Actual Image")
    
    test_emd=get_embeding_uploaded(model, uploaded_image_temp)
    test_emd_series=pd.Series(test_emd)

    df_embd_temp=df_embeded.copy()
    df_embd_temp=df_embd_temp.append(test_emd_series, ignore_index=True)
    
    # Calculate the cosine/jaccard distances
    cosine_distance_upload, indices_upload = calculate_cosine_distances(df_embd_temp)
    
    idx_ref = len(df_embd_temp)-1
    idx_rec, idx_sim = similarity_dataset(cosine_distance_upload , indices_upload , idx_ref)

    plt.imshow(uploaded_image_temp)
    figures = {'im'+str(i): load_image(DATASET_PATH,row.image_name) for i, row in df_images.loc[idx_rec].iterrows()}
    plot_images(figures, 2,2 )
    

# Run this for Recommendation from DATASET
def get_recommendation_from_dataset_image(idx, df_embeded, df_images):
    # Calculate the cosine/jaccard distances
    cosine_distance, indices = calculate_cosine_distances(df_embeded)
    
    plt.imshow(cv2.cvtColor(load_image(DATASET_PATH, df_images["image_name"].iloc[idx]), cv2.COLOR_BGR2RGB))
    plt.title("Actual Image")
    idx_rec, idx_sim = similarity_dataset(cosine_distance, indices, idx)

    figures = {'im'+str(i): load_image(DATASET_PATH,row.image_name) for i, row in df_images.loc[idx_rec].iterrows()}
    plot_images(figures, 2,2 )

    


# In[ ]:




