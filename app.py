import streamlit as st
import pickle
import pandas as pd

# Load the trained Logistic Regression model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature names used in the model
with open('features.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Function to preprocess the input data
def preprocess_input(features):
    # Convert categorical variables to dummy variables
    features = pd.get_dummies(features, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome'])
    
    return features

# Function to predict the target variable based on input features
def predict_target(features):
    prediction = model.predict(features)
    return prediction

# Main function to run the Streamlit application
def main():
    # Set the title and description of the app
    st.title("Bank Marketing Prediction")
    st.write("Enter the customer details to predict if they will subscribe to a term deposit or not.")

    # Create input fields for customer details
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                               'retired', 'self-employed', 'services', 'student', 'technician',
                               'unemployed', 'unknown'])
    marital = st.selectbox("Marital Status", ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("Default - has credit in default?",['no','yes','unknown'])
    housing = st.selectbox("Housing loan",['no','yes','unknown'])
    loan = st.selectbox("Personal loan",['no','yes','unknown'])
    poutcome = st.selectbox("Poutcome - outcome of the previous marketing campaign",['failure','other','success','unknown'])
    
    

    # Create a button to trigger the prediction
    if st.button("Predict"):
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default':[default],
            'housing':[housing],
            'loan':[loan],
            'poutcome':[poutcome]
        })

        # Preprocess the input data
        input_data_processed = preprocess_input(input_data)

        # Make the prediction
        prediction = predict_target(input_data_processed)

        # Display the prediction result
        if prediction[0] == 1:
            st.write("The customer is likely to subscribe to a term deposit.")
        else:
            st.write("The customer is unlikely to subscribe to a term deposit.")

# Run the main function
if __name__ == '__main__':
    main()
