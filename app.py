from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Tool 4 Image</h1>"
@app.route('/image')
def image():
    return "<h1>Image</h1>"