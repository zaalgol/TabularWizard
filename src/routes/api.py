from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask import jsonify, make_response
# from app.entities.model import Model
from src.services.inference_service import InferenceService
from src.services.training_service import TrainingService
# from app.services.token_serivce import TokenService
# from app.services.user_service import UserService
from flask_cors import CORS

from flask_jwt_extended import create_access_token, verify_jwt_in_request
# Create a Blueprint
bp = Blueprint('main', __name__)
CORS(bp)

# Instantiate UsersService singleton
# user_service = UserService()
inference_service = InferenceService()
training_Service = TrainingService()

@bp.route('/', methods=['GET'])
# @jwt_required()
def hello_world():
    return 'Hello, World!'

# call it with:
#  url = 'http://localhost:5000/api/trainModel/'
#         myobj = { 'model_dict': model.__dict__,'dataset': dataset}
#         # myobj = {'model': model, 'headers': df.columns.tolist(), 'df': df}

#         x = requests.post(url, json = myobj)
@bp.route('/api/trainModel/', methods=['POST'])
# @jwt_required()
def train_model():
    model_dict = request.json.get('model_dict', None)
    dataset = request.json.get('dataset', None)
    # app_context = request.json.get('app_context', None)
   
    # model = Model(user_id=user_id, model_name=model_name, description=description,
    #                model_type=model_type, training_strategy=training_strategy, sampling_strategy=sampling_strategy, target_column=target_column, metric=metric)

    training_result = training_Service.train_model(model_dict, dataset)
    return make_response(jsonify({"result": training_result}), 200)
