from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from joblib import load
import re
import string
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the saved SVM model
stopwords = nltk.corpus.stopwords.words('english')
loaded_model = load("svm_model.pkl") 
loaded_cv_vectorizer = load('cv.joblib')


ps = PorterStemmer()

# Preprocess text function
# def preprocess_text(text):
#     def clean_text(text):
#         text = "".join([word.lower() for word in text if word not in string.punctuation])
#         tokens = re.split('\W+', text)
#         text = [ps.stem(word) for word in tokens if word not in stopwords]
#         return ' '.join(text)

#     cleaned_text = clean_text(text)
#     return cleaned_text

@app.get("/", response_class=HTMLResponse)
async def get_form(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def post_form(request:Request, message: str = Form(...)):
    # print(f'message: {message}')
    # text = preprocess_text(message)
    # tfidf_vector = tfidf_vect.transform([text])
    
    new_txt = loaded_cv_vectorizer.transform([message])
    prediction = loaded_model.predict(new_txt)
    # print(prediction)
    prediction_probabilities = loaded_model.predict_proba(new_txt)
    spam_probability = prediction_probabilities[0][1] 
    # print(f"Probability of being spam: {spam_probability:.2%}")
    if prediction[0] == 'ham':
        prediction_result = "Not likely a Spam!"
    else:
        prediction_result = "Likely a Spam!"

    
    return templates.TemplateResponse("index.html", {"request": request, "predicted_result": prediction_result, "spam_probability": round(spam_probability,4)})

