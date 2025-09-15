# ms/__init__.py

from flask import Flask

# Create the Flask app instance
app = Flask(__name__)

# Import routes to register them with the app
from ms import routes