from flask import Flask
from flask_cors import CORS
from config import Config
from app.routes import model_routes, stream_routes

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    CORS(app)

    app.register_blueprint(model_routes)
    app.register_blueprint(stream_routes)

    return app
