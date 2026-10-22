import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


@st.cache_resource
def train():
    df = pd.read_table(URL, header=None, names=["label", "message"])
    df["y"] = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )
    vec = TfidfVectorizer(stop_words="english")
    model = MultinomialNB().fit(vec.fit_transform(X_train), y_train)
    acc = accuracy_score(y_test, model.predict(vec.transform(X_test)))
    return vec, model, acc, len(df)


st.set_page_config(page_title="Spam Message Checker", page_icon="📩")
st.title("📩 Spam Message Checker")
st.write("Type or paste an SMS message, and the model will tell you if it looks like spam.")

vec, model, acc, n_messages = train()

message = st.text_area("Message", placeholder="e.g. Congratulations! You won a free prize. Call now to claim")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please type a message first.")
    else:
        prob_spam = model.predict_proba(vec.transform([message]))[0][1]
        if prob_spam >= 0.5:
            st.error(f"🚨 Looks like SPAM ({prob_spam:.0%} spam probability)")
        else:
            st.success(f"✅ Looks like a normal message ({prob_spam:.0%} spam probability)")
        st.progress(float(prob_spam))

with st.expander("About this model"):
    st.write(
        f"Trained on {n_messages:,} SMS messages using TF-IDF and Naive Bayes. "
        f"Accuracy on messages it had not seen: {acc:.1%}. "
        "It only learns word patterns, so it can make mistakes, especially on new kinds of spam."
    )
