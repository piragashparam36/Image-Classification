import os 
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import numpy as np


input_dir = r'C:\Users\pirag\Documents\Projects\Image Classification\clf-data'
categories = ['empty' , 'not_empty']
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


data =[]
labels = []

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        if file.endswith(valid_extensions): 
            image_path = os.path.join(category_path, file)
            img = imread(image_path)
            img = resize(img, (15, 15)) 
            data.append(img.flatten())
            labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

#Test train split 
x_train, x_test , y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)

        
#Train Set Classifier 
classifier = SVC()

parameters = [{'gamma' : [0.01, 0.001, 0.0001] , 'C' : [1, 10, 100, 1000]}]  #12 Image classifier 
grid_search = GridSearchCV(classifier,parameters, cv=5)
grid_search.fit(x_train, y_train)


best_estimator = grid_search.best_estimator_

y_pridiction = best_estimator.predict(x_test)

score = accuracy_score(y_pridiction,y_test)

print('{} of samples were correctly classified'.format(str(score*100)))


pickle.dump(best_estimator, open('./modelIC.p','wb'))
