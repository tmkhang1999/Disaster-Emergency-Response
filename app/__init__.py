import logging
from flask import Flask

log = logging.getLogger(__name__)


def create_app():
    # Flask app setup
    app = Flask(__name__, template_folder="../app/templates")

    with app.app_context():
        from app.modules import main as main_blueprint
        app.register_blueprint(main_blueprint)

    return app
