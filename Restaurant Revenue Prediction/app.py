import streamlit as st
import pickle
from PIL import Image 
import pandas as pd

def main():
    st.set_page_config(layout='wide')
    st.title(':rainbow[Restaurant Revenue Prediction]')

    cuisine_images = {
        'Japanese': 'japanese.jpg',
        'Mexican': 'mexican.jpeg',
        'Italian': 'italian.jpeg',
        'Indian': 'indian.jpeg',
        'French': 'french.jpeg',
        'American': 'american.jpeg'
    }

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.subheader("Enter Restaurant Details:")
        Location = st.selectbox('Select the location of Restaurant:', ['Rural', 'Downtown', 'Suburban'])
        Cuisine = st.selectbox('Select Cuisine:', ['Japanese', 'Mexican', 'Italian', 'Indian', 'French', 'American'])
        st.image(cuisine_images[Cuisine], width=250)

    with col2:
        Seating_Capacity = st.slider('Seating Capacity', min_value=30, max_value=90, step=1)
        Average_Meal_Price = st.number_input("Average Meal Price ($)", min_value=5.0, max_value=200.0, step=1.0)
        Marketing_Budget = st.slider("Marketing Budget ($)", min_value=604, max_value=9978, step=1)
        Social_Media_Followers = st.slider("Social Media Followers", min_value=5277, max_value=103777, step=1)

    with col3:
        Chef_Experience_Years = st.number_input("Chef Experience Years", min_value=1, max_value=19, step=1)
        Ambience_Score = st.number_input("Ambience Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)
        Service_Quality_Score = st.number_input("Service Quality Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)
        Weekend_Reservations = st.slider('Weekend Reservations', min_value=0, max_value=88, step=1)
        Weekday_Reservations = st.slider('Weekday Reservations', min_value=0, max_value=88, step=1)

    features = pd.DataFrame([[Location, Cuisine, Seating_Capacity, Average_Meal_Price, Marketing_Budget, 
                               Social_Media_Followers, Chef_Experience_Years, Ambience_Score, Service_Quality_Score, 
                               Weekend_Reservations, Weekday_Reservations]],
                            columns=['Location', 'Cuisine', 'Seating_Capacity', 'Average_Meal_Price', 
                                     'Marketing_Budget', 'Social_Media_Followers', 'Chef_Experience_Years', 'Ambience_Score', 
                                     'Service_Quality_Score', 'Weekend_Reservations', 'Weekday_Reservations'])

    encoder = pickle.load(open('EL.sav', 'rb'))
    location_df = pd.DataFrame(encoder.transform(features[['Location']]), 
                               columns=encoder.get_feature_names_out())

    encoder1 = pickle.load(open('EC.sav', 'rb'))
    cuisine_df = pd.DataFrame(encoder1.transform(features[['Cuisine']]), 
                              columns=encoder1.get_feature_names_out())

    features = pd.concat([features, location_df, cuisine_df], axis=1)
    features.drop(['Location', 'Cuisine'], axis=1, inplace=True)

    model = pickle.load(open('model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))

    if st.button('PREDICT', use_container_width=True):
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        st.markdown("<h3 style='text-align: center; color: green;'>Predicted Revenue: ${:,.2f}</h3>".format(prediction[0]), unsafe_allow_html=True)


main()
