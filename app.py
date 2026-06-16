import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Function to create a download link for a Dataframe as a csv file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    Age = st.number_input("Age (years)", min_value=0, max_value=150)

    Sex = st.selectbox("Sex", ["Male", "Female"])

    Chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )

    Resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300
    )

    Cholesterol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=0
    )

    Fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        ["<= 120 mg/dl", "> 120 mg/dl"]
    )

    Resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )

    Max_hr = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=202
    )

    Exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Yes", "No"]
    )

    Oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0
    )

    St_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )


    # Convert categorical inputs to numeric
    Sex = 0 if Sex == "Male" else 1
    Chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(Chest_pain)
    Fasting_bs = 1 if Fasting_bs == "> 120 mg/dl" else 0
    Resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(Resting_ecg)
    Exercise_angina = 1 if Exercise_angina == "Yes" else 0
    St_slope = ["Upsloping", "Flat", "Downsloping"].index(St_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [Age],
        'Sex': [Sex],
        'ChestPainType': [Chest_pain],
        'RestingBP': [Resting_bp],
        'Cholesterol': [Cholesterol],
        'FastingBS': [Fasting_bs],
        'RestingECG': [Resting_ecg],
        'MaxHR': [Max_hr],
        'ExerciseAngina': [Exercise_angina],
        'Oldpeak': [Oldpeak],
        'ST_Slope': [St_slope]
    })

    algonames = ['Decision Tree','Logistic Regression','Support vector machine','Random Forest']
    modelnames = ['DT.pkl','LogisticR.pkl','SVM.pkl','RF.pkl']



    
    import os
    import pickle

    predictions = []

    def predict_heart_disease(data):
        

        base_dir = os.path.dirname(__file__)  # ensures correct file path

        for modelname in modelnames:
            model_path = os.path.join(base_dir, modelname)

            with open(model_path, "rb") as file:
                model = pickle.load(file)

            prediction = model.predict(data)
            predictions.append(prediction)

        return predictions

    # Create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('-------------------------------------')

        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("No heart disease detected.")
            else:
                st.write("Heart disease detected.")
            st.markdown('-------------------------------------')


    with tab2:
        st.title("Upload CSV File")

        st.subheader('Instructions to note before uploading the file:')
        st.info("""
            1. No NaN values allowed.
            2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
            3. Check the spellings of the feature names.
            4. Feature values conventions: \n
                - Age: age of the patient [years] \n
                - Sex: sex of the patient [0: Male, 1: Female] \n
                - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic] \n
                - RestingBP: resting blood pressure [mm Hg] \n
                - Cholesterol: serum cholesterol [mm/dl] \n
                - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] \n
                - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria] \n
                - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] \n
                - ExerciseAngina: exercise-induced angina [1: Yes, 0: No] \n
                - Oldpeak: oldpeak = ST [Numeric value measured in depression] \n
                - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping] \n
    """)
        
        # Create a file uploader in the sidebar
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded CSV file into a DataFrame
            #input_data = pd.read_csv(uploaded_file)
            try:
    # 1. First attempt: Read standard comma-separated file
                input_data = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Fallback for encoding issues (e.g., Excel exports)
                uploaded_file.seek(0)
                input_data = pd.read_csv(uploaded_file, encoding='latin1')
            except pd.errors.ParserError:
                try:
                    # 2. Second attempt: Check if the file uses Semicolons (;) instead of commas
                    uploaded_file.seek(0)
                    input_data = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
                except pd.errors.ParserError:
                    # 3. Third attempt: If there's broken header metadata, skip bad lines or look for content
                    uploaded_file.seek(0)
                    input_data = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='latin1')

            model = pickle.load(open('LogisticR.pkl','rb'))

            # Ensure that the input DataFrame matches the expected columns and format
            expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

            if set(expected_columns).issubset(input_data.columns):
                input_data['Prediction LR'] = ''

                for i in range(len(input_data)):
                    arr = input_data.iloc[i,:-1].values
                    input_data['Prediction LR'][i] = model.predict([arr])[0]
                input_data.to_csv('PredictedHeartLR.csv')

                # Display the predictions
                st.subheader("Predictions:")
                st.write(input_data)

                # Create a button to download the updated CSV file
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
            else:
                st.warning("Please make sure the uploaded CSV file has the correct columns.")

        else:
            st.info("Upload a CSV file to get predictions.")

    with tab3:
        import plotly.express as px
        data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22}
        Models = list(data.keys())
        Accuracies = list(data.values())
        df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
        fig = px.bar(df,y='Accuracies',x='Models')
        st.plotly_chart(fig)