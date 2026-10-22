# Spam Message Checker (Web App)

A web app where you type an SMS message and a machine learning model tells you if it looks like spam.

**Live app:** https://spam-checker-web-app-avuwlbe4ackpu8trwj39cp.streamlit.app/

- **Model:** TF-IDF features and Naive Bayes, trained when the app starts
- **Dataset:** SMS Spam Collection (5,572 messages), the same data as my Spam-message-detection project
- **Accuracy:** about 97% on messages the model had not seen
- **Built with:** Python, scikit-learn, Streamlit

## How to use
1. Open the live app link.
2. Type or paste an SMS message.
3. Click **Check message** to see whether it looks like spam, with a spam probability.

## Run it yourself
```
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
The model only learns word patterns from this dataset, so it can miss new kinds of spam. The app may take a few seconds to wake up if nobody has used it for a while.
