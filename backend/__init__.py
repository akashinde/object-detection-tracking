from flask import Flask
from flask_cors import CORS

from .db import ensure_normalized_db_tables, UPLOAD_FOLDER, PROCESSED_FOLDER
from .routes import bp


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

    ensure_normalized_db_tables()
    app.register_blueprint(bp)
    return app
