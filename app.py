from crypt import methods
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
sgd_clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():

	messagess = request.form['message']
	data = [messagess]
	vect = cv.tranform(data).toarray()
	my_prediction = sgd_clf.predict(vect)

	return render_template('result.html',prediction = my_prediction)
	

# @app.route('/predict',methods=['POST'])
# def predict():
    
#     if request.method == 'POST':
		
# 		message = request.form['message']
# 		data = [message]
# 		vect = cv.transform(data).toarray()
# 		my_prediction = sgd_clf.predict(vect)
# 	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)