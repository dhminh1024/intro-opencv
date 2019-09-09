from blueprints import *
from flask import Flask

app = Flask(__name__)
app.register_blueprint(homepage)
app.register_blueprint(upload)

app.debug = True

# App Settings
app.config['UPLOAD_FOLDER'] = 'uploads'

if __name__ == '__main__':
  app.run(debug=True)