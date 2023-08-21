import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


# Load the dataset (replace 'your_data.csv' with your actual file)
df = pd.read_csv('Food_CAL.csv')

# Convert categorical features to numerical using one-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['type', 'core']])

# Combine all features into a feature matrix
features = df[['calories']]
features = pd.concat([features, pd.DataFrame(encoded_features)], axis=1)

# Calculate cosine similarity between food items
cosine_sim = cosine_similarity(features, features)

# Function to get top N food recommendations based on given features
def get_food_recommendations(query_type, query_core, query_calories, top_n=5):
    query_encoded = encoder.transform([[query_type, query_core]])
    query_features = [query_calories] + list(query_encoded[0])
    
    similarity_scores = cosine_similarity([query_features], features)
    similar_food_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    similar_food_info = df.iloc[similar_food_indices]
    return similar_food_info

# Streamlit app
st.title("Food Recommendation System")

# User inputs
calories = st.slider("Select Calories", min_value=0, max_value=500, step=10)
food_type = st.radio("Select Food Type", df['type'].unique())

# Determine available food core values based on food type
if food_type == 'Vegetarian':
    available_cores = ['Aloo', 'Coffee', 'Dal', 'Dosa', 'Ice Cream', 'Khichdi', 'Naan', 'Paneer', 'Paratha', 'Roti', 'Sabzi', 'Tea']
else:
    available_cores = ['Chicken', 'Egg', 'Fish', 'Mutton']

food_core = st.selectbox("Select Food Core", available_cores)

# Get recommendations on button click
if st.button("Get Recommendations"):
    with st.empty():
        st.subheader("Recommended Foods:")
        recommendations = get_food_recommendations(food_type, food_core, calories)
        pd.set_option('display.width', 1000)  # Set the desired width
        st.dataframe(recommendations[['food_name', 'quantity', 'calories', 'type', 'core', 'carbohydrates(g)', 'fat(g)', 'protein(g)']])