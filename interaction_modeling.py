# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:19:24 2019

@author: sapkotab
"""

import numpy as np
import pandas as pd
import re
import os
import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import from_levels_and_colors
import cv2
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier




#function to train the model using trainng data in csv file
def train_model(traindata, test_size, epochs, batch_size):
    
    #dataset import
    dataset = pd.read_csv(traindata) #You need to change #directory accordingly

    #checking for null values
    if dataset.isnull().values.any(): print("Missing data found") 
    else: print("No missing data found")

    #Normalizing the data
    scaler = MinMaxScaler()
    scaler.fit(dataset.iloc[:, 1:])
    normalized=scaler.transform(dataset.iloc[:,1:])
    data=pd.concat([pd.DataFrame(normalized), dataset.iloc[:,0]], axis=1)

    ##Splitting dataframes
    X_train,X_test,y_train,y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],test_size = 0.2, stratify=data.iloc[:, -1])

    ##Converting the y labels to one-hot encoding
    ohe = OneHotEncoder()
    y_train = ohe.fit_transform(np.array(y_train.tolist()).reshape(-1,1)).toarray()
    y_test = ohe.fit_transform(np.array(y_test.tolist()).reshape(-1,1)).toarray()

    ### Training the model#####
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Training the model
    traininit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    #Prediction on test data
    y_pred = model.predict(X_test)

    #Converting predictions to label and one-hot encoded reference test label to label
    pred = [np.argmax(y_pred[i]) for i in range(len(y_pred))]
    test = [np.argmax(y_test[i]) for i in range(len(y_test))]

    #Accuracy assessment
    pred, test=np.asarray(pred), np.asarray(test)

    acc = classification_report(test, pred, output_dict=True)
    acc=pd.DataFrame.from_dict(acc)

    return model, acc, scaler


###### function to inference over image using the model trained above
def inference_over_image (img, scaler, model, outputfilename):
    
    #loading the image
    img=cv2.imread(img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Constructing features
    R=img[:,:,0].astype(float)
    G=img[:,:,1].astype(float)
    B=img[:,:,2].astype(float)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]
    EXG=((2*G)-R-B)/(R+G+B)
    VARI= (G-R)/ (G+R-B)

    #using scaler constructed during training to normalize the image data
    layers=np.array([R,G,B,h,s,v,EXG,VARI])
    res=layers.reshape((layers.shape[1]*layers.shape[2], layers.shape[0]))
    res=scaler.transform(res)

    #Predicting with the model
    result = model.predict(res)

    #converting the result elements into user assigned label form
    pred = [np.argmax(result[i]) for i in range(len(result))]

    #Converting list of labels into array with shapes equal to original image
    pred_arr= np.asanyarray(pred)
    pred_arr= pred_arr.reshape((layers.shape[1], layers.shape[2]))

    #Displaying the final classification results using user defined color map.
    from matplotlib.colors import from_levels_and_colors
    cmap, norm = from_levels_and_colors([0,1,2, 3],['yellow','blue','red'])
    plt.imshow(pred_arr, cmap=cmap, norm=norm)
    cv2.imwrite(outputfilename, pred_arr)
    print("Classified map written successfully")



####### function to convert the classified map to grid maps
def classify_2_heatmap (classified_imgpath, class_of_interest, gridsize, eqn, eqn_type, percentc, savemappath):
    
    #assigning variables to some of the the user-fed parameters
    n=int(gridsize)
    classid=class_of_interest

    #loading the classified image in its original form
    img=cv2.imread(classified_imgpath, -1)
    a,b = img.shape[0]%n, img.shape[1]%n

    #trimming the image to the dimension exactly divisible by n
    if a==0:
        x=img.shape[0]
    else:
        x=img.shape[0] - a

    if b==0:
        y=img.shape[1]
    else:
        y=img.shape[1] - b

    clip_img=img[0:x, 0:y]

    #calculating the number of grids in terms of nrows and ncols.
    nrows=int(clip_img.shape[0]/n)
    ncols=int(clip_img.shape[1]/n)

    #finding the column positions for the all the grids
    colidx=[]
    idx=0
    for t in range(ncols):
        idx=idx+n
        idx1 = idx - 1
        colidx.append(idx1)

    #finding the row positions for the all the grids
    rowidx=[]
    idx=0
    for t in range(nrows):
        idx=idx+n
        idx1=idx-1
        rowidx.append(idx1)
    colidx.insert(0, 0)
    rowidx.insert(0, 0)

    #finding the frequency of the class of interest in each of the grids determined above
    freq=[]
    count = 0
    for i in range(len(rowidx)):
        for j in range(len(colidx)):
            count=0
            if i<len(rowidx)-1 and j<len(colidx)-1:
                for row in range(rowidx[i], rowidx[i+1]):
                    for col in range(colidx[j], colidx[j+1]):
                        # print(row)
                        if clip_img[row, col]==classid:
                            count+=1
            else:
                break
            freq.append(count)

    #Converting list of frequency to array and reshaping to the nrows and ncols
    freq_arr=np.array(freq)
    freq_arr=freq_arr.reshape((nrows, ncols))

    #solve path 
    if savemappath is not None:
        savemappath1=os.path.join(savemappath, "eqn_map.jpg")
        savemappath2 = os.path.join(savemappath, "percent_coverage.jpg")
        savemappath3 = os.path.join(savemappath, "combined.jpg")

    # Solving the equation fed by user where y would be the response variable user is looking
    # for and x would be the array of frequency obtained above
    if eqn:
        if eqn_type =="simple_linear":
            text=eqn
            parts= re.findall(r"[0-9.]+|.", text)
            parts=list(filter(None, parts))
            parts=' '.join(parts).split()
            if len(parts)==8:
                m=float(parts[3])
                c=float(parts[7])
                m=-m
                if parts[6]=="-":
                    freq_arr1 = freq_arr * m - c
                    labeltext = "biomass(g) per grid"
                else:
                    freq_arr1 = freq_arr * m + c
                    labeltext = "biomass(g) per grid"
            else:
                m=float(parts[2])
                c=float(parts[6])
                if parts[5]=="-":
                    freq_arr1 = freq_arr * m - c
                    labeltext = "biomass(g) per grid"
                else:
                    freq_arr1 = freq_arr * m + c
                    labeltext = "biomass(g) per grid"

            if savemappath:
                fig1, ax= plt.subplots(1,1)
                heatmap=ax.imshow(freq_arr1, cmap=plt.cm.Reds)
                cbar = fig1.colorbar(heatmap, ax=ax)
                cbar.set_label(labeltext)
                fig1.savefig(savemappath1)
                print("The map is saved successfully")

    #converting the array of frequency to the percentage cover if user assigns percentc=True.
    if percentc:
        freq_arr2=freq_arr/ (n*n)*100
        labeltext1="% coverage per grid"
        fig2, ax = plt.subplots(1, 1)
        heatmap = ax.imshow(freq_arr2, cmap=plt.cm.Reds)
        cbar = fig2.colorbar(heatmap, ax=ax)
        cbar.set_label(labeltext1)
        fig2.savefig(savemappath2)
        print("The map is saved successfully")
    else:
        pass

    #plotting the maps together if conditions are met
    if eqn and percentc:
        fig3, ax = plt.subplots(2, 2, figsize=(15,15))
        ax[0,0].imshow(img)
        heatmap1 = ax[0,1].imshow(freq_arr, cmap=plt.cm.Reds)
        cbar = fig3.colorbar(heatmap1, ax=ax[0,1])
        cbar.set_label("# of pixels per grid ")
        heatmap2 = ax[1, 0].imshow(freq_arr2, cmap=plt.cm.Reds)
        cbar = fig3.colorbar(heatmap2, ax=ax[1,0])
        cbar.set_label(labeltext1)
        heatmap3 = ax[1,1].imshow(freq_arr1, cmap=plt.cm.Reds)
        cbar = fig3.colorbar(heatmap3, ax=ax[1,1])
        cbar.set_label(labeltext)
        fig3.savefig(savemappath3)
        print("The map is saved successfully")


if __name__=="__main__":
    img="sample.tif"
    traindata = "trainingsamples.csv"
    model, accuracy_summary, scaler = train_model(traindata, 0.3, 500, 20)
    inference_over_image(img, scaler, model, "test_classified.tif")
    classify_2_heatmap ("test_classified.tif", 1, 20, False, 'simple_linear', True, os.getcwd())
