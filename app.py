from flask import Flask, render_template, request
import pandas as pd
from Pipelines import setup_prediction_pipeline

app = Flask(__name__)
app.secret_key = 'fdgdfg45e45e4tgd'

prediction_pipeline = setup_prediction_pipeline(
    target_column_name="X",
    max_length=10,
    tokenizer_path=r"tokenizer2.pkl",
    label_encoder_path=r"\label_encoder2.pkl",
    unique_words_path=f"",
    model_path=r"\ChatBotIntentsModel3.h5",
    intents_path=r"intents4.json"
)


def response(user_msg):
    df = pd.DataFrame({"X": [user_msg]})
    df = prediction_pipeline.fit_transform(df)

    return df["Answers"][0]


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(response(userText))


if __name__ == '__main__':
    app.run("127.0.0.1", port=8000, debug=True)
