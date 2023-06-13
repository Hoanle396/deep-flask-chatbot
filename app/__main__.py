from app import app
import app.settings as settings
from flask_frozen import Freezer
from controllers.route import registryRouter
from models.model import ChatBot

model = ChatBot()

if __name__ == "__main__":
    registryRouter(app, model)
    freezer = Freezer(app)
    freezer.freeze()
