from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from pydantic import BaseModel
import pandas as pd
app = Flask(__name__)

model = pickle.load(open('LGBM.pkl', 'rb'))
model1 = pickle.load(open('LGBM1.pkl', 'rb'))
model2 = pickle.load(open('LGBM2.pkl','rb'))


class Scoringitem(BaseModel):
    
    Age:int
    Gender_Female: bool
    Gender_Male : bool
    Polyuria_No : bool
    Polyuria_Yes: bool
    Polydipsia_No: bool
    Polydipsia_Yes : bool
    suddenweightloss_No: bool
    suddenweightloss_Yes : bool
    weakness_No: bool
    weakness_Yes: bool
    Polyphagia_No: bool
    Polyphagia_Yes: bool
    Genitalthrush_No: bool
    Genitalthrush_Yes: bool
    visualblurring_No: bool
    visualblurring_Yes: bool
    Itching_No: bool
    Itching_Yes: bool
    Irritability_No: bool
    Irritability_Yes: bool
    delayedhealing_No : bool
    delayedhealing_Yes: bool
    partialparesis_No: bool
    partialparesis_Yes: bool
    musclestiffness_No: bool
    musclestiffness_Yes: bool
    Alopecia_No: bool
    Alopecia_Yes: bool
    Obesity_No : bool
    Obesity_Yes: bool
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    data = request.form.to_dict()
    print("Form Data:", data)

    # Ensure all features are present
    for feature in Scoringitem.__annotations__.keys():
        if feature not in data:
            return render_template('home.html', prediction_text="Missing feature: {}".format(feature))

    # Convert integer values to bool
    for key, value in data.items():
        if(key == 'Age'):
            data[key]= int(value)
        if key != 'Age':  # 'Age' should remain as int
            data[key] = bool(int(value))
        
    final_features = pd.DataFrame([data])
    yhat1 = model.predict(final_features)
    yhat2 = model1.predict(final_features)
    yhat3 = model2.predict(final_features)
    yhat = (yhat1 + yhat2 + yhat3) / 3.0

    return render_template('home.html', prediction_text="Do you have diabetes: {}".format(yhat[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Extract numerical values from JSON data
        data = [float(x) for x in data.values()]

        # Create a DataFrame
        final_features = pd.DataFrame([data], columns=[ 'Age' , 'Gender_Female',
    'Gender_Male ',
    'Polyuria_No ' ,
    'Polyuria_Yes' ,
    'Polydipsia_No' ,
    'Polydipsia_Yes ', 
    'suddenweightloss_No' ,
    'suddenweightloss_Yes ', 
    'weakness_No' ,
    'weakness_Yes' ,
    'Polyphagia_No' ,
    'Polyphagia_Yes' ,
    'Genitalthrush_No', 
    'Genitalthrush_Yes', 
    'visualblurring_No', 
    'visualblurring_Yes', 
    'Itching_No', 
    'Itching_Yes' ,
    'Irritability_No', 
    'Irritability_Yes', 
    'delayedhealing_No ', 
    'delayedhealing_Yes', 
    'partialparesis_No' ,
    'partialparesis_Yes', 
    'musclestiffness_No', 
    'musclestiffness_Yes', 
    'Alopecia_No', 
    'Alopecia_Yes', 
    'Obesity_No ' ,
    'Obesity_Yes' 
    ])

        # Make predictions
        prediction = model.predict_proba(final_features)

        # Convert the NumPy array to a Python list
        prediction_list = prediction[0].tolist()

        # Ensure that the list only contains JSON serializable types
        for item in prediction_list:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

        # Return the prediction in JSON format
        return jsonify({"prediction_probability": prediction_list},{"prediction": int(prediction_list[0])})
               
    

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
