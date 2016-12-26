################################################################################
###############################  IMPORTING LIBRARIES  ##########################
################################################################################

import pandas as pd
import glob
import re
import numpy as np
import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix 
from sklearn.svm import SVC
from mahotas.features import surf
from sklearn.cluster import KMeans
import pickle
import graphlab as gl

################################################################################
###############################  DEFINING FUNCTIONS  ###########################
################################################################################

def loadingimages(folderpath):
    image_list = []
    filename_list=[]

    for filename in glob.glob(folderpath): #assuming gif
        filename_list.append(filename)
        image_list.append(mh.imread(filename))

    filename_list[:] = [re.sub('C.*?k\\\\','',s) for s in filename_list]
    return image_list, filename_list

def gettinglabels(labelxl,filename_list):
    label = [] 
    for f in filename_list:
        label.append(re.sub('.*?c','c',str(labelxl.loc[labelxl['img']==f].classname)))
    label[:]=[s[:-4] for s in label]
    return label        

def descriptors(image_list):
    alldescriptors = []
    for image in image_list:
        image = mh.colors.rgb2gray(image, dtype=np.uint8)
        alldescriptors.append(surf.dense(image, spacing=16))
    return alldescriptors    

def concdescrp(alldescriptors):
    # get all descriptors into a single array
    concatenated = np.concatenate(alldescriptors)
    print('Number of descriptors: {}'.format(len(concatenated)))
    # use only every 64th vector
    concatenated = concatenated[::64]    
    return concatenated
    
def km_model_pkl(k,concatenated):
    km = KMeans(k)
    km_model = km.fit(concatenated)
    output = open('km_model.pkl', 'wb')
    pickle.dump(km_model, output)
    output.close()

def BoVW(alldescriptors,km_model,k):
    sfeatures = []
    for d in alldescriptors:
        c = km_model.predict(d)
        sfeatures.append(np.array([np.sum(c == ci) for ci in range(k)]))
    # build single array and convert to float
    sfeatures = np.array(sfeatures, dtype=float)
    return sfeatures
    
def classifier_pkl(model,outputfname):
    output = open(outputfname, 'wb')
    pickle.dump(model, output)
    output.close()

def predictfn(clf,features,correctlabel):
    predictedclass = clf.predict(features)
    predictedprob = clf.predict_proba(features)
    accuracy = accuracy_score(correctlabel,predictedclass)
    logloss = log_loss(correctlabel,predictedprob)
    print "Accuracy: ",accuracy,"\n"
    print "Log loss: ",logloss, "\n"
    print confusion_matrix(correctlabel,predictedclass)
    return predictedprob



################################################################################
################################ MAIN FUNCTION #################################
################################################################################


####################### LOAD TRAIN DATASET ###########################
folderpath= 'C:/Users/Nishaat/Desktop/Statistica Learning/Data Stats learning/train8k/*.jpg'
#Load images and their names
train_images, train_filenames = loadingimages(folderpath)
#Get labels for training images
labelxl = pd.read_csv('C:\Users\Nishaat\Desktop\Statistica Learning\Data Stats learning\driver_imgs_list.csv')
train_labels = gettinglabels(labelxl,train_filenames)


####################### EXTRACTING FEATURES ###########################
#GET DESCRIPTORS OF TRAIN IMAGES
train_alldescriptors = descriptors(train_images)

#CONCATENATE ALL TRAIN IMAGES' DESCRIPTORS
concatenated = concdescrp(train_alldescriptors)

#RUN K-MEANS CLUSTERING TO FIND VISUAL WORDS
km_model_pkl(256,concatenated)

#FIND FEATURES OF TRAIN IMAGES
#Get the KM model
pkl_file = open('km_model.pkl', 'rb')
km = pickle.load(pkl_file)
pkl_file.close()
#Get features as a bag of visual words
features_all_train = BoVW(train_alldescriptors,km,256)


####################### FITTING THE MODELS ###########################
#RUN CLASSIFICATION ALGORITHM
#Logistic Regression
logreg=LogisticRegression()
LR=logreg.fit(features_all_train,train_labels)
#Pickle the model
classifier_pkl(LR,'LRclf.pkl')

#Multiclass SVM
mc_svm_clf = SVC(decision_function_shape='ovr',probability=True)
mc_svm_model = mc_svm_clf.fit(features_all_train,train_labels)
#Pickle the model
classifier_pkl(mc_svm_model,'MCSVMclf.pkl')

#Graphlab Create - DT algorithms
data = gl.SFrame(features_all_train)
target = gl.SArray(train_labels)
data.add_column(target)
gl_model = gl.classifier.create(data, target='X2')

#GL Random Forest
glrfclf = gl.random_forest_classifier.create(data,'X2')

#GL BoostedTree
glbtclf = gl.boosted_trees_classifier.create(data,'X2')


#################### TRAINING ACCURACY & LOGLOSS #####################
##Logistic Regression
pkl_file = open('LRclf.pkl', 'rb')
LRclf = pickle.load(pkl_file)
pkl_file.close()

predictfn(LRclf,features_all_train,train_labels)

#MCSVM
pkl_file = open('MCSVMclf.pkl', 'rb')
MCSVMclf = pickle.load(pkl_file)
pkl_file.close()

predictfn(MCSVMclf,features_all_train,train_labels)

#GL
results = gl_model.evaluate(data)
results_rf = glrfclf.evaluate(data)
results_bt = glbtclf.evaluate(data)


########################## LOAD TEST DATASET ########################
folderpath= 'C:/Users/Nishaat/Desktop/Statistica Learning/Data Stats learning/test1k/*.jpg'
#Load images and their names
test_images, test_filenames = loadingimages(folderpath)
#Get labels for training images
test_labels = gettinglabels(labelxl,test_filenames)

####################### EXTRACTING TEST FEATURES #####################
#GET DESCRIPTORS OF TEST IMAGES
test_alldescriptors = descriptors(test_images)

#CONCATENATE ALL TEST IMAGES' DESCRIPTORS
concatenated = concdescrp(test_alldescriptors)

#FIND FEATURES OF TEST IMAGES
#Get the KM model
pkl_file = open('km_model.pkl', 'rb')
km = pickle.load(pkl_file)
pkl_file.close()
#Get features as a bag of visual words
features_all_test = BoVW(test_alldescriptors,km,256)


#################### TESTING ACCURACY & LOGLOSS #####################
##Logistic Regression
pkl_file = open('LRclf.pkl', 'rb')
LRclf = pickle.load(pkl_file)
pkl_file.close()

predictfn(LRclf,features_all_test,test_labels)

#MCSVM
pkl_file = open('MCSVMclf.pkl', 'rb')
MCSVMclf = pickle.load(pkl_file)
pkl_file.close()

predictedprob = predictfn(MCSVMclf,features_all_test,test_labels)

#GL

test_data = gl.SFrame(features_all_test)
test_target = gl.SArray(test_labels)
test_data.add_column(test_target)

results1 = gl_model.evaluate(test_data)
test_results = glrfclf.evaluate(test_data)
test_results_bt = glbtclf.evaluate(test_data)

#Final results - TO BE SUBMITTED IN KAGGLE
final_result = pd.DataFrame()
final_result['ImageName'] = test_filenames
final_result['c0']=[i[0] for i in predictedprob]
final_result['c1']=[i[1] for i in predictedprob]
final_result['c2']=[i[2] for i in predictedprob]
final_result['c3']=[i[3] for i in predictedprob]
final_result['c4']=[i[4] for i in predictedprob]
final_result['c5']=[i[5] for i in predictedprob]
final_result['c6']=[i[6] for i in predictedprob]
final_result['c7']=[i[7] for i in predictedprob]
final_result['c8']=[i[8] for i in predictedprob]
final_result['c9']=[i[9] for i in predictedprob]


######################### Cross Validation #################################
cv = cross_validation.KFold(len(features_all_train), 5, shuffle=True, random_state=123)

#Logistic Regression
scores1 = cross_validation.cross_val_score(LRclf, features_all_train, train_labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores1.mean()))

#SVM
scores2 = cross_validation.cross_val_score(MCSVMclf, features_all_train, train_labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores2.mean()))



