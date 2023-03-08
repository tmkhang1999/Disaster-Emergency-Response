from app import create_app
from models.ml_pipeline import tokenize

if __name__ == '__main__':
    # Initialize app
    app = create_app()

    # Running
    app.run(debug=True, host='0.0.0.0', port='8080', threaded=True)
