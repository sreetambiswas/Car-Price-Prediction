#CAR PRICE PREDICTION - FLASK WEB APP (DEPLOYMENT)


from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model artifacts
model           = joblib.load("outputs/car_price_model.pkl")
scaler          = joblib.load("outputs/scaler.pkl")
label_encoders  = joblib.load("outputs/label_encoders.pkl")
selected_feats  = joblib.load("outputs/selected_features.pkl")
all_feat_cols   = joblib.load("outputs/all_feature_cols.pkl")

# ---- HTML template ----
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Car Price Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
  .card {
    background: white;
    border-radius: 20px;
    padding: 40px;
    max-width: 700px;
    width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
  }
  h1 {
    font-size: 26px;
    color: #1a1a2e;
    margin-bottom: 6px;
    font-weight: 700;
  }
  .subtitle { color: #666; font-size: 14px; margin-bottom: 28px; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  label {
    display: block;
    font-size: 13px;
    color: #333;
    font-weight: 600;
    margin-bottom: 4px;
  }
  input, select {
    width: 100%;
    padding: 10px 12px;
    border: 1.5px solid #e0e0e0;
    border-radius: 8px;
    font-size: 14px;
    color: #222;
    transition: border-color 0.2s;
    outline: none;
  }
  input:focus, select:focus { border-color: #0f3460; }
  .btn {
    margin-top: 24px;
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #0f3460, #533483);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
  }
  .btn:hover { opacity: 0.88; }
  .result {
    margin-top: 24px;
    background: linear-gradient(135deg, #0f3460, #533483);
    color: white;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .result .label { font-size: 13px; opacity: 0.8; margin-bottom: 4px; }
  .result .price { font-size: 36px; font-weight: 800; letter-spacing: 1px; }
  .error {
    margin-top: 20px;
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-radius: 10px;
    padding: 14px;
    color: #b91c1c;
    font-size: 14px;
  }
  .section-label {
    font-size: 11px;
    font-weight: 700;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 10px;
  }
</style>
</head>
<body>
<div class="card">
  <h1>&#x1F697; Car Price Predictor</h1>
  <p class="subtitle">Enter car specifications to get an estimated market price</p>

  <form method="POST" action="/predict">
    <p class="section-label">Engine & Performance</p>
    <div class="grid">
      <div>
        <label>Engine Size (cc)</label>
        <input type="number" name="enginesize" placeholder="e.g. 130" required value="{{ vals.enginesize or '' }}">
      </div>
      <div>
        <label>Horsepower</label>
        <input type="number" name="horsepower" placeholder="e.g. 102" required value="{{ vals.horsepower or '' }}">
      </div>
      <div>
        <label>Fuel System</label>
        <select name="fuelsystem">
          {% for opt in fuel_opts %}
          <option value="{{ opt }}" {% if vals.fuelsystem == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Fuel Type</label>
        <select name="fueltype">
          {% for opt in fueltype_opts %}
          <option value="{{ opt }}" {% if vals.fueltype == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <p class="section-label">Body & Dimensions</p>
    <div class="grid">
      <div>
        <label>Car Body</label>
        <select name="carbody">
          {% for opt in carbody_opts %}
          <option value="{{ opt }}" {% if vals.carbody == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Drive Wheels</label>
        <select name="drivewheel">
          {% for opt in drive_opts %}
          <option value="{{ opt }}" {% if vals.drivewheel == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Curb Weight (lbs)</label>
        <input type="number" name="curbweight" placeholder="e.g. 2548" required value="{{ vals.curbweight or '' }}">
      </div>
      <div>
        <label>Car Length (inches)</label>
        <input type="number" step="0.1" name="carlength" placeholder="e.g. 168.8" required value="{{ vals.carlength or '' }}">
      </div>
      <div>
        <label>Car Width (inches)</label>
        <input type="number" step="0.1" name="carwidth" placeholder="e.g. 64.1" required value="{{ vals.carwidth or '' }}">
      </div>
      <div>
        <label>Wheel Base (inches)</label>
        <input type="number" step="0.1" name="wheelbase" placeholder="e.g. 98.8" required value="{{ vals.wheelbase or '' }}">
      </div>
    </div>

    <p class="section-label">Efficiency & Other</p>
    <div class="grid">
      <div>
        <label>City MPG</label>
        <input type="number" name="citympg" placeholder="e.g. 26" required value="{{ vals.citympg or '' }}">
      </div>
      <div>
        <label>Highway MPG</label>
        <input type="number" name="highwaympg" placeholder="e.g. 30" required value="{{ vals.highwaympg or '' }}">
      </div>
      <div>
        <label>Compression Ratio</label>
        <input type="number" step="0.1" name="compressionratio" placeholder="e.g. 9.0" required value="{{ vals.compressionratio or '' }}">
      </div>
      <div>
        <label>Number of Cylinders</label>
        <select name="cylindernumber">
          {% for opt in cyl_opts %}
          <option value="{{ opt }}" {% if vals.cylindernumber == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Car Company</label>
        <select name="CarCompany">
          {% for opt in company_opts %}
          <option value="{{ opt }}" {% if vals.CarCompany == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label>Engine Type</label>
        <select name="enginetype">
          {% for opt in engine_opts %}
          <option value="{{ opt }}" {% if vals.enginetype == opt %}selected{% endif %}>{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <button class="btn" type="submit">&#x1F4B0; Predict Car Price</button>
  </form>

  {% if predicted_price %}
  <div class="result">
    <div class="label">Estimated Market Price</div>
    <div class="price">${{ predicted_price }}</div>
  </div>
  {% elif error %}
  <div class="error">&#x26A0;&#xFE0F; {{ error }}</div>
  {% endif %}
</div>
</body>
</html>
"""

# Default dropdown options (from the training dataset)
OPTIONS = {
    "fuel_opts":     ["mpfi", "2bbl", "mfi", "1bbl", "spfi", "4bbl", "idi", "spdi"],
    "fueltype_opts": ["gas", "diesel"],
    "carbody_opts":  ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
    "drive_opts":    ["fwd", "rwd", "4wd"],
    "cyl_opts":      ["four", "six", "five", "eight", "two", "three", "twelve"],
    "company_opts":  sorted([
        "toyota", "honda", "mazda", "nissan", "mitsubishi", "volkswagen",
        "subaru", "bmw", "audi", "mercedes-benz", "volvo", "jaguar",
        "chevrolet", "dodge", "plymouth", "buick", "porsche", "alfa-romero",
        "peugeot", "saab", "isuzu", "renault"
    ]),
    "engine_opts":   ["ohc", "ohcv", "ohcf", "dohc", "rotor", "l", "dohcv"]
}


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, vals={}, **OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Build input dict with all required columns (use median defaults for missing)
        input_data = {
            'symboling':        0,
            'wheelbase':        float(form['wheelbase']),
            'carlength':        float(form['carlength']),
            'carwidth':         float(form['carwidth']),
            'carheight':        50.0,
            'curbweight':       float(form['curbweight']),
            'enginesize':       float(form['enginesize']),
            'boreratio':        3.19,
            'stroke':           3.40,
            'compressionratio': float(form['compressionratio']),
            'horsepower':       float(form['horsepower']),
            'peakrpm':          5200.0,
            'citympg':          float(form['citympg']),
            'highwaympg':       float(form['highwaympg']),
            'fueltype':         form['fueltype'],
            'aspiration':       'std',
            'doornumber':       'four',
            'carbody':          form['carbody'],
            'drivewheel':       form['drivewheel'],
            'enginelocation':   'front',
            'enginetype':       form['enginetype'],
            'cylindernumber':   form['cylindernumber'],
            'fuelsystem':       form['fuelsystem'],
            'CarCompany':       form['CarCompany'],
        }

        # Encode categorical fields
        for col in label_encoders:
            if col in input_data:
                le = label_encoders[col]
                val = input_data[col]
                # Handle unseen labels gracefully
                if val in le.classes_:
                    input_data[col] = le.transform([val])[0]
                else:
                    input_data[col] = 0

        # Create engineered features
        cyl_map = {'two':2,'three':3,'four':4,'five':5,'six':6,'eight':8,'twelve':12}
        cyl_num = cyl_map.get(form['cylindernumber'], 4)
        input_data['power_to_weight'] = float(form['horsepower']) / float(form['curbweight'])
        input_data['displacement_per_cyl'] = float(form['enginesize']) / cyl_num
        input_data['avg_mpg'] = (float(form['citympg']) + float(form['highwaympg'])) / 2

        # Build DataFrame with all features in correct order
        df_input = pd.DataFrame([input_data])

        # Ensure all feature columns are present
        for col in all_feat_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[all_feat_cols]

        # Scale
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=all_feat_cols)

        # Select features
        df_final = df_scaled[selected_feats]

        # Predict
        price = model.predict(df_final)[0]
        price = max(price, 0)  # no negative prices

        return render_template_string(
            HTML,
            predicted_price=f"{price:,.0f}",
            vals=form,
            error=None,
            **OPTIONS
        )

    except Exception as e:
        return render_template_string(
            HTML,
            predicted_price=None,
            vals=request.form,
            error=str(e),
            **OPTIONS
        )


if __name__ == "__main__":
    print("\n  Car Price Prediction App")
    print("  Running at: http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)