import azure.functions as func
import logging
from flask import Flask,request
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the saved model
with open('masdar_ticket_model_as_group.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CountVectorizer
with open('masdar_ticket_transformer_as_group.pkl', 'rb') as f:
    count_vect = pickle.load(f)

def get_top_k_predictions_with_probabilities(model, X_test, k):
    # Get probabilities instead of predicted labels
    probs = model.predict_proba(X_test)
    
    # Get top k predictions by probability indices
    best_n = np.argsort(probs, axis=1)[:, -k:]
    
    # Get categories and probabilities of top k predictions
    top_k_preds_with_probs = []
    for indices, prob_row in zip(best_n, probs):
        top_k_preds_with_probs.append([(model.classes_[index], prob_row[index]) for index in indices[::-1]])
    
    return top_k_preds_with_probs

def predict_classification(text):
    # Preprocess the input text using the same CountVectorizer used during training
    text_vectorized = count_vect.transform([text])
    
    # Predict the classification
    predicted_class = model.predict(text_vectorized)
    
    return predicted_class[0]

def predict_classification_scores(text):
    # Preprocess the input text using the same CountVectorizer used during training
    text_vectorized = count_vect.transform([text])
    
    # Predict the classification
    predicted_class = model.predict(text_vectorized)
    predicted_score = model.predict_proba(text_vectorized)

    out = get_top_k_predictions_with_probabilities(model,text_vectorized,k=1)
    # print(out)
    return out

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="get_group")
def get_group(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the JSON data from the request
    data = request.get_json()
    print(data)    
    predict = predict_classification_scores(data['name'])
    if predict[0][0][0] is not None:        
        output = {"status":"success","category": predict[0][0][0],"score":predict[0][0][1]}
        # return {"status":"success","category": predict[0][0][0],"score":predict[0][0][1]}, 200
        func.HttpResponse(f"Hello, {output}.",status_code=200)
    else:
        # return {"status":"Some error occurred"}
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    # else:
    #     return func.HttpResponse(
    #          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
    #          status_code=200
    #     )