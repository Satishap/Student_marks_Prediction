from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load(
    '/home/satish/Machine Learning/Student Marks Prediction.pkl')
df = pd.DataFrame()

# Create your views here.


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global df
    input_feature = [int(x) for x in request.form.values()]
    feature_value = np.array(input_feature)

    
    if input_feature[0] < 0 or input_feature[0] >24:
        return render_template("index.html", Predict_text="Abe pagal Ek Din Me kitna Ghanta Hota hai  Pahle Pad Ke aa ")
    output = model.predict([feature_value])[0][0].round(2)
    df=pd.concat([df,pd.DataFrame({'Study Hours':input_feature, 'Predicted Output':[output]})], ignore_index=True)
    print(df)
    df.to_csv('hhhh.csv')
    return render_template('index.html',Predict_text="You will be get [{}%] marks, When You do  study [{}] hours per day " .format(output, int(feature_value[0])))



if __name__ == "__main__":
    app.run()
