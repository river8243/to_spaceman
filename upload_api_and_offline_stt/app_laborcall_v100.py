from api_service_stt_v100 import Operation
import os
import sys
from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from laborcall_framework_v100 import EbVoicePlatformAPI
sys.path.append(os.getcwd())

operation_method = Operation()
app = Flask(__name__)
CORS(app, resource=r'/*')
app.config['JSON_AS_ASCII'] = False
api = Api(app)
api.add_resource(
    EbVoicePlatformAPI,
    '/if_stt/stt',
    endpoint='/if_stt/stt',
    resource_class_kwargs={'method': operation_method}
)

if __name__ == '__main__':
    app.run(debug=False)
