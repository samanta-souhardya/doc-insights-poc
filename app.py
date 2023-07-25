from flask_cors import CORS
from flask import Flask
from dotenv import dotenv_values
from index_route import indexRoutes

config = dotenv_values(".env")
print(config)

app = Flask(__name__)

CORS(app)

app.register_blueprint(indexRoutes)


if __name__ == '__main__':
    app.run()