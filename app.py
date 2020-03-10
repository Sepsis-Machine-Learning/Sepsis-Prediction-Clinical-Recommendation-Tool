#imports for website
from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from io import StringIO

#imports for machine learning
import tensorflow as tf
from tensorflow import keras
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split

#configures app and the database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test2.db'
db = SQLAlchemy(app)

#each patient is a Todo class
class Todo(db.Model):
    content = db.Column(db.String(200), nullable=False)
    weight = db.Column(db.Float())
    id = db.Column(db.Integer(), primary_key=True)
    patient_id = db.Column(db.String())
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return '<Task %r>' % self.id

#entering values into the database and updating the table
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        task_weight = request.form['weight']
        task_patient_id = request.form['patient_id']
        new_task = Todo(content=task_content, weight = task_weight, patient_id = task_patient_id )
        print (new_task)
        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except Exception as error:
            print (error)
            return 'There was an issue adding your data'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html', tasks=tasks)

#search bar
@app.route('/search/', methods=['GET'])
def search():
    task_search = request.args['search']
    tasks = Todo.query.filter(Todo.patient_id.like(task_search)).all()
    return render_template('searchr.html', tasks=tasks)

#delete function
@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that data'

#update function
# @app.route('/update/<int:id>', methods=['GET', 'POST'])
# def update(id):
#     task = Todo.query.get_or_404(id)

#     if request.method == 'POST':
#         task.content = request.form['content']

#         try:
#             Todo.session.commit()
#             return redirect('/')
#         except:
#             return 'There was an issue updating your task'

#     else:
#         return render_template('update.html', task=task)

#initializes the loaded up model
def create_model(n_cols):
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n_cols,)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    keras.optimizers.Adam(lr = 0.001)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

#generates the results from the model and the kmeans clustering
@app.route('/results/<int:id>', methods={'GET', 'POST'})
def results(id):
    #normalizes the data around 1
    def NormalizeData(data):
         return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    #queries the patient Todo class that was requestted 
    task = Todo.query.get_or_404(id)

    #creates the model with pretrained and saved weights
    model = create_model(40)
    model.load_weights('./weights/my_weights')
    #model = keras.models.load_model("model1")
    #global model
    #print(task.content)
    #test = pd.read_csv(StringIO("HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel\n58|96|36.5|131|87|72|17|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|12|NaN|9.3|NaN|0.71|NaN|103|NaN|1.9|NaN|3.9|NaN|NaN|36.2|11.8|NaN|5.7|NaN|201|82|0|NaN|NaN|-128.39|21|0"), sep='|')
    #test = pd.read_csv("setB_test.psv", sep='|')

    #converts the patient data String into a csv
    titles = "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel\n" + task.content
    print (titles)
    test = pd.read_csv(StringIO(titles), sep='|')
    
    #changes hospital admit time into a positive number
    test.HospAdmTime = -1 * test.HospAdmTime

    #turning the values from csv input into floats to be evaluated
    kmessage = ""
    HR = float(test.HR.values)
    Age = float(test.Age.values)
    ICULOS = float(test.ICULOS.values)
    HospAdmTime = -1 * float(test.HospAdmTime.values)
    Gender = float(test.Gender.values)

    #checks to see if the values are within range of high risk groups
    print (str(HR) + " " + str(Age) + " " + str(Gender) + " " + str(ICULOS) + " " + str(HospAdmTime))
    if 86 <= HR <= 89 and  68 <= Age <= 72 and Gender <= .5 and 19 <= ICULOS <= 23 and 33 <= HospAdmTime <= 37:
        kmessage = "By Kmeans Clustering utilizing 4 clusters, this patient is in a group that is around 10 times more likely to get sepsis"
    if 74 < HR <= 75 & 66 <= Age <= 67 & Gender <= .5 &  17 <= ICULOS <= 18 & 8 <= HospAdmTime <= 9:
        kmessage = "By Kmeans Clustering utilizing 6 clusters, this patient is in a group that is around 10 times more likely to get sepsis"
    if 87 <= HR <= 88 & 60 <= Age <= 61 & 35 <= HospAdmTime <= 36 & 10 <= ICULOS <= 11:
        kmessage = "By Kmeans Clustering utilizing 5 clusters, this patient is in a group that is around 10 times more likely to get sepsis"


    #preprocessing for the model
    #test=test.drop(test.columns[cols],axis=1)
    #replace NaN with mean
    test[np.isnan(test)] = -99
    test=test.values
    x_test_raw = test[:, :-1]
    y_test = test[:, -1]  # for last column 
    x_test = NormalizeData(x_test_raw)

    #classifications = model.predict(x_test)
    #test_loss, test_acc = model.evaluate(x_test, y_test)    

    #pushes specific data point into the predict funciton
    y_predict_probs = model.predict(x_test, verbose=0)
    y_predict_classes = model.predict_classes(x_test, verbose=0)

    y_probs = y_predict_probs[:, 0]
    #y_classes = y_predict_classes[:, 0]

    #print("predict"+str(y_probs))
    
    #changes the float outputed by the machine learning model into a number on a scale 1-100
    score_float = float("".join(map(str, y_probs)))
    score_float = score_float*100
    score_float = round(score_float, 5)
    print(score_float)

    #calculates the amount of bolus based on the patients weight needed to be injected
    bolus_calc = task.weight*0.453592*30
    bolus_calc = round(bolus_calc,3)
    
    return render_template('results.html', result = str(score_float), tpat= task, bc = bolus_calc, mess = kmessage)


if __name__ == '__main__':
    #import matplotlib.pyplot as plt

    # train = pd.read_csv("setB_train.psv", sep='|')
    cols = [7]
    # # print(train.shape)
    # train.HospAdmTime = -1 * train.HospAdmTime
    # train=train.drop(train.columns[cols],axis=1)
    # train[np.isnan(train)] = -99
    # train=train.values

    # x_train_raw = train[:, :-1] # for all but last column
    # n_rows, n_cols = x_train_raw.shape

    # y_train = train[:, -1]  # for last column

    # global model
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(n_cols,)),
    #     keras.layers.Dense(10, activation=tf.nn.relu),
    #     keras.layers.Dense(1, activation=tf.nn.sigmoid),
    # ])
    # keras.optimizers.Adam(lr = 0.001)
    # model.compile(optimizer='adam',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])
    
    # x_train = NormalizeData(x_train_raw)
    # model.fit(x_train, y_train, epochs=3)
    #model.save("model1")
    app.run(debug = True)
