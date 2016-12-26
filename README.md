# Image-Classification--Statefarm-Kaggle
Predicting if the driver is distracted or not based on dashboard cam images

One in five car accidents is caused by a distracted driver. This is an alarming statistic, even more so for car insurers. Profits would take a toll every time such a customer made a claim. State Farm is giving dashboard cams a try to check if it would help in these situations. If the footage can show that the driver was distracted at the time of an accident, State Farm could save money in those one in five cases. In this project, we will be detecting if a driver is distracted or not based on the images obtained from dashboard cams using machine learning algorithms.

Algorithm used for - 

image processing: SURF
feature extraction: Bag of visual words (using k-means on image descriptors)
image classification: Logistic Regression and SVM

How to run the code:
Packages required:
Pickle: used to export the python model into a function
Mahotas: image feature extraction
Scikit learn: for performing machine learning
Re, numpy, pandas, and os: for other operations

Run the final_code file first to get the pickle files for k-means clusters and the classification model. Before running this file, make sure that all the folder paths have been correctly specified and that they end in a ‘k’.
Use these pickle files in the FinalDemo  and make the following changes

Paths:
 In the line: 
app=Flask(__name__, template_folder=’ please input the folder path of templates here’). This line will connect the templates web html file with the python file, which will get the input and show result in the webpage.

In this line:
app.config['UPLOAD_FOLDER'] = 'please input the folder path that you want to save the image uploaded(could be anywhere you want’).This path saves the files you upload in the website.

In this line: labelxl = pd.read_csv('/Users/hao/PycharmProjects/BIA656_Final_Demodriver_imgs_list.csv') please change the path into the path you store the csv file.

Run:
Make sure you installed all the package and set all the path correctly. Run the FinalDemo.py in console, it will shows the address of localhost, click that address to visit this web application.
