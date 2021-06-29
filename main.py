from flask import Flask
from flask_ask import Ask, statement, question
import alexa_yolo
app=Flask(__name__)
ask=Ask(app, '/')

@ask.launch
def start_skill():
    welcome_message = " hey do you want help to see?"
    return question(welcome_message)

@ask.intent("yesIntent")
def yesIntent():
    labels = alexa_yolo.AngelEye()
    text ="the detected object are"
    for label in labels:
        text = text+", "+label
    print(text)
    return statement(text)

if __name__=="__main__" :
    app.run(port=7045, debug=True)
