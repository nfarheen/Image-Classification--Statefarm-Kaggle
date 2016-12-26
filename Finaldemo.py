# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:27:09 2016

@author: Nishaat
"""

import pickle
import mahotas as mh
from mahotas.features import surf
import re
import numpy as np
import pandas as pd


import os
import json
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, Response

# Initialize the Flask application
app = Flask(__name__, template_folder = 'C:\Users\mohit\Desktop\Spring BIA\Stats Learning\Project\democode')

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'C:\Users\mohit\Desktop\Spring BIA\Stats Learning\Project\democode'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def hello_world():
    return render_template('index.html')
# Route that will process the file upload
    
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Move the file form the temporal folder to
        # the upload folder we setup
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result = classifier(filepath)
        jsondata = [{"id":"logistic", "classes":result}]
        #  #Convert the JSON data into a Python structure
        #data = json.loads(jsondata)
        return render_template('index.html', data=jsondata)
        # return redirect(url_for('json_result', result = result))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file

#testing one image demo
def classifier(filename):
    #filename = 'C:/Users/Nishaat/Desktop/Data Stats learning/test1k/img_101810.jpg'
    pkl_file = open('km_model.pkl', 'rb')
    km = pickle.load(pkl_file)
    pkl_file.close()
    
    #filename = 'C:/Users/Nishaat/Desktop/Data Stats learning/test1k/img_101810.jpg'
    test_image = mh.imread(filename)
    image = mh.colors.rgb2gray(test_image, dtype=np.uint8)
    descriptor = surf.dense(image, spacing=16)
    filename = re.sub('C.*?1k\/','',filename)
    
    labelxl = pd.read_csv('C:\Users\mohit\Desktop\Spring BIA\Stats Learning\Project\driver_imgs_list.csv')
    label1 = re.sub('.*?c','c',str(labelxl.loc[labelxl['img']==filename].classname))
    label1=label1[:-4]
    
    k=256
    c = km.predict(descriptor)
    sfeatures1 = np.array([np.sum(c == ci) for ci in range(k)])
    
    # build single array and convert to float
    sfeatures_test1 = np.array(sfeatures1, dtype=float)
    
    pkl_file = open('LRclf.pkl', 'rb')
    LR = pickle.load(pkl_file)
    pkl_file.close()
           
    predict_x1=LR.predict(sfeatures_test1)
        #prob1=logreg.predict_proba(sfeatures_test1)
        #pd.crosstab(label_test,predict_x,rownames=['True'],colnames=['Predicted'],margins=True)
        
    return predict_x1[0]    

if __name__ == '__main__':
    app.run()

