import pickle 
import pandas as pd
from fastapi import FastAPI
import uvicorn  # ASGI

# Load the model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API! Use the /predict endpoint to make predictions."}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: dict):
    latitude = data['latitude']
    longitude = data['longitude']
    price = data['price']
    minimum_nights = data['minimum_nights']
    reviews_per_month = data['reviews_per_month']
    calculated_host_listings_count = data['calculated_host_listings_count']
    availability_365 = data['availability_365']
    number_of_reviews_ltm = data['number_of_reviews_ltm']
    neighbourhood_group_Eixample = data['neighbourhood_group_Eixample']
    neighbourhood_group_Gràcia = data['neighbourhood_group_Gràcia']
    neighbourhood_group_Horta_Guinardó = data['neighbourhood_group_Horta-Guinardó']
    neighbourhood_group_Les_Corts = data['neighbourhood_group_Les Corts']
    neighbourhood_group_Nou_Barris = data['neighbourhood_group_Nou Barris']
    neighbourhood_group_Sant_Andreu = data['neighbourhood_group_Sant Andreu']
    neighbourhood_group_Sant_Martí = data['neighbourhood_group_Sant Martí']
    neighbourhood_group_Sants_Montjuïc = data['neighbourhood_group_Sants-Montjuïc']
    neighbourhood_group_Sarrià_Sant_Gervasi = data['neighbourhood_group_Sarrià-Sant Gervasi']
    room_type_Hotel_room = data['room_type_Hotel room']
    room_type_Private_room = data['room_type_Private room']
    room_type_Shared_room = data['room_type_Shared room']
    neighbourhood_target_encoded = data['neighbourhood_target_encoded']

    # Convert input data into a DataFrame
    input_data = pd.DataFrame([data])

    # Make predictions
    prediction = model.predict(input_data)

    return {"prediction": prediction.tolist()}

# Run the app with: uvicorn app:app --reload
