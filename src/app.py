from flask import Flask, request, render_template
from pickle import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = load(open("decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Sin Diabetes",
    "1": "Con Diabetes"
}

scaler = StandardScaler()

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Obtain values from form
        preg = float(request.form["preg"])
        gluc = float(request.form["gluc"])
        bloodpress = float(request.form["bloodpress"])
        skinthick = float(request.form["skinthick"])
        ins = float(request.form["ins"])
        bmi = float(request.form["bmi"])
        DiaPedFun = float(request.form["DiaPedFun"])
        age = float(request.form["age"])
        
        data = scaler.transform([preg, gluc, bloodpress, skinthick, ins, bmi, DiaPedFun, age])

        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)
