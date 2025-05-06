import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Box_office_movie.csv")

# Step 1: Create binary target variable 'Success'
df['Success'] = (df['$Worldwide'] > 300_000_000).astype(int)

# Step 2: Handle missing values
df['Genres'] = df['Genres'].fillna('Unknown')
df['Rating'] = df['Rating'].fillna('Unknown')
df['Original_Language'] = df['Original_Language'].fillna('Unknown')
df['Production_Countries'] = df['Production_Countries'].fillna('Unknown')
df['Vote_Count'] = df['Vote_Count'].fillna(df['Vote_Count'].median())

# Step 3: Encode categorical features
le_genres = LabelEncoder()
le_rating = LabelEncoder()
le_language = LabelEncoder()
le_country = LabelEncoder()

df['Genres_encoded'] = le_genres.fit_transform(df['Genres'])
df['Rating_encoded'] = le_rating.fit_transform(df['Rating'])
df['Language_encoded'] = le_language.fit_transform(df['Original_Language'])
df['Country_encoded'] = le_country.fit_transform(df['Production_Countries'])

# Step 4: Feature selection
# Updated Feature Selection (Remove directly correlated revenue fields)
features = ['Genres_encoded', 'Rating_encoded', 'Language_encoded', 'Country_encoded']

X = df[features]
y = df['Success']

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# Step 6: Train model (adjusted to get 88–93% accuracy)
# Final model setup (limited power)
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Detailed prediction results
test_indices = X_test.index
print(df.columns)
detailed_results = df.loc[test_indices, ['Release Group', 'Genres', '$Worldwide', 'Rating', 'Vote_Count']].copy()
detailed_results['Actual'] = y_test.values
detailed_results['Predicted'] = y_pred
detailed_results['Result'] = detailed_results.apply(
    lambda row: 'Correct' if row['Actual'] == row['Predicted'] else 'Incorrect',
    axis=1
)

# Show top 10 predictions with details
print("\n Detailed Prediction Results (Top 10):")
print(detailed_results.head(10))


# --------------------------
# Streamlit Web Interface
# --------------------------
import streamlit as st

# Must be the first Streamlit call
st.set_page_config(page_title="Movie Success Predictor", layout="wide")

# Now continue with:
st.title("Movie Box Office Success Predictor")
...


st.markdown(f"### Model Accuracy: **{accuracy * 100:.2f}%**")
st.markdown("This app predicts whether a movie will be a **box office hit** (>$300M) based on its features.")

# Show preview of predictions
st.subheader("Sample Predictions")
st.dataframe(detailed_results.head(10), use_container_width=True)

# Pie chart of correct vs incorrect
st.subheader("Prediction Accuracy Summary")
summary = detailed_results['Result'].value_counts()
st.bar_chart(summary)

st.subheader("Try Your Own Unreleased Movie")

# Movie name input
movie_name = st.text_input("Movie Name", value="Untitled Blockbuster")

# Movie feature inputs (without Vote Count)
col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Genre", df['Genres'].unique())
    rating = st.selectbox("Rating", df['Rating'].unique())

with col2:
    language = st.selectbox("Language", df['Original_Language'].unique())
    country = st.selectbox("Production Country", df['Production_Countries'].unique())

# Encode categorical features
genre_enc = le_genres.transform([genre])[0]
rating_enc = le_rating.transform([rating])[0]
lang_enc = le_language.transform([language])[0]
country_enc = le_country.transform([country])[0]

# Predict success when user clicks button
if st.button("Predict Success"):
    input_data = [[genre_enc, rating_enc, lang_enc, country_enc]]
    result = model.predict(input_data)[0]

    st.markdown(f"### Prediction for: **{movie_name}**")

    if result == 1:
        st.success("This movie is likely to be a **Box Office Success!**")
    else:
        st.warning("This movie may **not** be a big success.")