from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string 

import pickle

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

model_LogReg = pickle.load(open("LogReg.pkl", "rb"))
model_RandForest = pickle.load(open("RandForest.pkl", "rb"))
model_DecTree = pickle.load(open("DecTree.pkl", "rb"))

vectorization = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input')

        testing_news = {"text":[user_input]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt) 
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)

        prediction_LogReg = model_LogReg.predict(new_xv_test)[0]
        if prediction_LogReg == 1: 
            prediction_LogReg = 'True News'
        else:
            prediction_LogReg = 'Fake News'

        prediction_RandForest = model_RandForest.predict(new_xv_test)[0]
        if prediction_RandForest == 1: 
            prediction_RandForest = 'True News'
        else:
            prediction_RandForest = 'Fake News'

        prediction_DecTree = model_DecTree.predict(new_xv_test)[0]
        if prediction_DecTree == 1: 
            prediction_DecTree = 'True News'
        else:
            prediction_DecTree = 'Fake News'

        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: auto;
                        min-height: 100vh;
                        background: linear-gradient(to right, #f2c94c, #f2994a);
                        color: white;
                    }
                    .form-container {
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                    }
                    .form-group, .btn {
                        width: 300px;
                    }
                    textarea {
                        height: 150px;
                    }
                </style>
            </head>
            <body>
                <div class="form-container">
                    <h1 class="text-center mb-4">FAKE OR REAL ?</h1>
                    <form method="POST" class="text-center">
                        <div class="form-group">
                            <textarea name="user_input" class="form-control" placeholder="Bir metin girin..."></textarea>
                        </div>
                        <button type="submit" class="btn btn-light">Gönder</button>
                    </form>
                    <p class="mt-2 text-center">Sonuç_LogReg : {{prediction_LogReg}} </p>
                    <p class="mt-2 text-center">Sonuç_RandForest : {{prediction_RandForest}} </p>
                    <p class="mt-2 text-center">Sonuç_DecTree : {{prediction_DecTree}} </p>
                </div>
            </body>
            </html>
        """, prediction_LogReg = prediction_LogReg, prediction_RandForest=prediction_RandForest,prediction_DecTree = prediction_DecTree)
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: auto;
                    min-height: 100vh;
                    background: linear-gradient(to right, #f2c94c, #f2994a);
                    color: white;
                }
                .form-container {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                }
                .form-group, .btn {
                    width: 300px;
                }
                textarea {
                    height: 150px;
                }
            </style>
        </head>
        <body>
            <div class="form-container">
                <h1 class="text-center mb-4">FAKE OR REAL ?</h1>
                <form method="POST" class="text-center">
                    <div class="form-group">
                        <textarea name="user_input" class="form-control" placeholder="Bir metin girin..."></textarea>
                    </div>
                    <button type="submit" class="btn btn-light">Gönder</button>
                </form>
            </div>
        </body>
        </html>
    """)

if __name__ == '__main__':
    app.run(debug=True)
