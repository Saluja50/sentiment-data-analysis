import streamlit as st
import joblib
import re
import emoji


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------
# Styling
# -----------------------------

st.markdown("""
<style>

.stApp {
    background: #080b14;
    color: white;
}

.block-container {
    max-width: 900px;
    padding-top: 4rem;
}

.hero {
    text-align: center;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 52px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #94a3b8;
    font-size: 17px;
}

.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 20px;
    padding: 25px;
}

.result {
    text-align: center;
    background: #111827;
    border-radius: 20px;
    padding: 35px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():

    return joblib.load("model.pkl")


model = load_model()


# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text):

    text = str(text).lower()

    text = emoji.demojize(
        text,
        delimiters=(" ", " ")
    )

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# -----------------------------
# Header
# -----------------------------

st.markdown("""
<div class="hero">

<h1>🧠 SentimentAI</h1>

<p>
Understand the sentiment behind any piece of text.
</p>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# Input
# -----------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

text = st.text_area(
    "Enter your text",
    placeholder="Example: I absolutely loved this movie! 😍",
    height=160
)

analyze = st.button(
    "✨ Analyze Sentiment",
    use_container_width=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Prediction
# -----------------------------
# -----------------------------
# Prediction
# -----------------------------

if analyze:

    if not text.strip():

        st.warning("Please enter some text.")

    else:

        cleaned = clean_text(text)

        prediction = model.predict(
            [cleaned]
        )[0]

        if isinstance(prediction, str):
            sentiment = prediction.capitalize()

        else:
            labels = {
                0: "Negative",
                1: "Neutral",
                2: "Positive"
            }

            sentiment = labels[int(prediction)]

        if sentiment == "Positive":
            icon = "😊"

        elif sentiment == "Negative":
            icon = "😞"

        else:
            icon = "😐"

        st.html(f"""
<div class="result">
    <div style="color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:2px;">
        Predicted Sentiment
    </div>
    <div style="font-size:42px;font-weight:800;margin-top:10px;">
        {icon} {sentiment}
    </div>
</div>
""")