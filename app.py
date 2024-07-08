from flask import Flask, render_template, request
import pandas as pd
import pickle
from io import StringIO

app = Flask(__name__)

# Load the model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample data for testing (replace this with your actual dataset)
sample_data = {
    'Average_Price': [100],
    'Advertising_Expenditure': [200],
    'Website_Traffic': [300],
    'Customer_Retention': [0.8],
    'Average_Order_Value': [50],
    'Seasonality': [1.5],
    'Regular_Customer_Value': [100],
    'Seasonal_Demand_Factor': [1.2],
    'Discount_Offer': [0],
    'Net_Revenue': [500]  # Replace 'Net_Revenue' with your actual target column name
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Assuming 'Net_Revenue' is your target column
target_column = 'Net_Revenue'

# Fit the model with sample data
model.fit(sample_df.drop(target_column, axis=1), sample_df[target_column])

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use StringIO from the io module
        input_data = pd.read_csv(StringIO(request.form['input_data']))
        
        # Make sure the model is fitted before making predictions
        predicted_values = model.predict(input_data)

        # Display the result
        return render_template('index.html', prediction=predicted_values)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
