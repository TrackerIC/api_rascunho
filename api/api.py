from flask import Flask
import jsonify
app = Flask(__name__)

@app.route("/prev")
def prev():
    return 'pato'



app.run()
