import joblib
import neattext.functions as nfx


model = joblib.load("../models/emotion_model.joblib")
vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")


def clean_text(text):
    text = text.lower()
    text = nfx.remove_stopwords(text)
    text = nfx.remove_punctuations(text)
    text = nfx.remove_urls(text)
    text = nfx.remove_special_characters(text)
    return text


def predict_emotion(text_input):
    cleaned_input = clean_text(text_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)
    proba = model.predict_proba(vectorized_input) if hasattr(model, "predict_proba") else None
    return prediction[0], proba


if __name__ == "__main__":
    user_text = input("Enter a sentence to detect emotion: ")
    emotion, probabilities = predict_emotion(user_text)
    print(f"Predicted Emotion: {emotion}")


    if probabilities is not None:
        print("\nPrediction Probabilities:")
        classes = model.classes_
        for cls, prob in zip(classes, probabilities[0]):
            print(f"  {cls}: {prob:.2f}")
