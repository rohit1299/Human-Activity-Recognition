from flask import Flask, render_template, request, jsonify
import pickle

import pickle
import numpy as np
from matplotlib import image as img
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
import pyttsx3

app= Flask(__name__)
model = pickle.load(open("HAR_Model.pkl",'rb'))
engine = pyttsx3.init()
df = pd.read_csv('C:/Users/Hp/PycharmProjects/HAR/Training_set.csv')
df= df.loc[(df["label"] == "sitting")|(df["label"] == "sleeping")| (df["label"] ==  "clapping")|  (df["label"] ==  "running")| (df["label"] ==  "eating")]
empty= pd.DataFrame()
pd.concat([empty,df], ignore_index=True)

# Label encoding and seperate dependant variable

lb = LabelBinarizer()
y = lb.fit_transform(df['label'])
classes = lb.classes_
print(classes)

# Function to read images as array

def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))

# Function to predict

def test_predict(test_image):
    result = model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = classes[itemindex[1][0]]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)
    engine.setProperty('rate', 100)
    engine.say(prediction)
    # engine.runAndWait()
    # engine.startLoop(False)
    # engine.endLoop()
    return prediction



@app.route('/')

def home():

    return render_template('index.html')
   
@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['file']

    result=test_predict(file)
    # show the result !
    Emotion= ("PREDITED  EMotion is : ", result)
    print(Emotion)

    classification = "Predicted Action: "+str(result)

    return render_template('index.html', prediction=classification)
    # engine.runAndWait()

if __name__ == '__main__':
     app.run(port=3000,debug=True)