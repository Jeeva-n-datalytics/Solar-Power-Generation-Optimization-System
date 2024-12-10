import streamlit as st
import pandas as pd
import joblib
from sklearn.base import TransformerMixin
import seaborn as sns

class AutoClean(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cleaned_data = X.dropna()  # Remove rows with missing values
        cleaned_data = cleaned_data.drop_duplicates()  # Remove duplicate rows
        # Assuming "Defective/Non Defective" is the target column and other columns are features
        features = ["Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "_If", "Vabc", "Vf"]
        cleaned_data = cleaned_data[features + ["Defective / Non Defective"]]
        return cleaned_data

def main():
    st.title(":sunny: SOLAR POWER GENERATION PREDICTION MODEL")
    st.sidebar.title(":sunny: Solar Power")

    html_temp = """
    <div style="background-color: tomato; padding: 10px">
    <h2 style="color: white; text-align: center;"> PV Defect & Power Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except:
            try:
                data = pd.read_excel(uploaded_file)
            except:      
                data = pd.DataFrame()
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color: tomato; padding: 10px">
    <p style="color: white; text-align: center;">Add Database Credentials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html=True)
            
    user = st.sidebar.text_input("user", "root")
    pw = st.sidebar.text_input("password", "root")
    db = st.sidebar.text_input("database", "dsproject")
    
    result = ""
    
    if st.button("Predict"):
        # Load the trained RandomForestClassifier
        classifier = joblib.load('best_random_forest_model.pkl')
        
        # Preprocess input data using AutoClean pipeline
        auto_clean_pipeline = joblib.load('AutoClean_pipeline.pkl')
        input_df_processed = auto_clean_pipeline.transform(data)
        
        # Make predictions
        predictions = classifier.predict(input_df_processed.drop("Defective / Non Defective", axis=1))
        
        # Display predictions in a table format
        result_df = pd.DataFrame({
            'Row Number': range(1, len(predictions) + 1),
            'Prediction': ['Defective' if pred == 1 else 'Non-Defective' for pred in predictions],
            'Original Values': input_df_processed['Defective / Non Defective'].values
        })
        
        # Apply background gradient to the table
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result_df.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()
