import os
import werkzeug
import json
# export local_test=1
# if not os.getenv('local_test'):
#    import uwsgi
import time
import uuid
from flask import make_response, jsonify, request
from flask_restful import Resource, reqparse

model = None


class EbVoicePlatformAPI(Resource):
    """
    API Interface
    output:probability
    """

    def __init__(self, method):
        self.method = method
        if request.method == 'POST':
            print(request.form)
            self.reqparse = reqparse.RequestParser(bundle_errors=True)
            self.reqparse.add_argument(
                'business_unit',
                type=str,
                required=True,
                help='No business_unit Provided!',
                location='form')
            self.reqparse.add_argument(
                'client_id',
                type=str,
                required=True,
                choices=['test_client_id'],
                help='Unauthorized client_id!',
                location='form')
            self.reqparse.add_argument(
                'request_id',
                type=str,
                required=True,
                help='No request_id Provided!',
                location='form')
            self.reqparse.add_argument(
                'api_version',
                type=str,
                required=False,
                help='No api_version Provided!',
                location='form')
            self.reqparse.add_argument(
                'inputs',
                type=str,
                required=False,
                help='No inputs Provided!',
                location='form')
            self.reqparse.add_argument(
                'file',
                type=werkzeug.datastructures.FileStorage,
                location='files')
        super().__init__()

    def post(self):
        start_time = time.time()
        unit_tags = uuid.uuid4()
        nginx_uuid = 'local_test'
        if not os.getenv('local_test'):
            nginx_uuid = request.headers.get('X_Request_Id')
            #uwsgi.set_logvar('request_id', nginx_uuid)
        self.method.logger.extra = {
            "project": "if_stt", "request_id": unit_tags,
            "worker": "api", "release_id": os.environ['releaseID']}
        # argss = self.bodyreqparse.parse_args(req=self.args)
        self.args = self.reqparse.parse_args(strict=True)
        argss = json.loads(self.args['inputs'])

        print('### args:', self.args)
        print('### argss:', argss)
        print('### file:', self.args['file'])
        if argss:
            self.method.logger.info({"flask_msg": "API start",
                                     "nginx_id": nginx_uuid,
                                     "request_id": self.args['request_id']})
            self.method.business_unit = self.args['business_unit']
            self.method.request_id = self.args['request_id']
            self.method.client_id = self.args['client_id']
            self.method.trace_id = unit_tags
            self.method.request_time = start_time
            self.method.file = self.args['file']
        else:
            self.method.logger.error(
                {"flask_msg": "Input error"}, exc_info=True)
            return make_response(
                jsonify({'error_msg': 'Get unexpected error with inputs!'}), 500)
        try:
            #self.method.logger.error({"flask_msg":"Enter"}, exc_info=True)
            argss = self.method.type_translate(argss)
            #self.method.logger.error({"flask_msg":"A"}, exc_info=True)
            if model:
                result = self.method.run(argss, model)
            else:
                result = self.method.run(argss)
            #self.method.logger.error({"flask_msg":"B"}, exc_info=True)
        except BaseException:
            self.method.logger.error(
                {"flask_msg": "result is unexpected!"}, exc_info=True)
            return make_response(jsonify({'error_msg': 'compute error!'}), 500)
        if result.keys():
            end_time = time.time()
            duration_time = round((end_time - start_time), 4)
            self.method.logger.info(
                {"flask_msg": "API end", "nginx_id": nginx_uuid, "duration_time": duration_time})
            output = {
                'business_unit': self.args['business_unit'],
                'request_id': self.args['request_id'],
                'trace_id': str(unit_tags),
                'request_time': start_time,
                'response_time': end_time,
                'duration_time': duration_time,
                'outputs': result}
            output_log = {
                'business_unit': self.args['business_unit'],
                'request_id': self.args['request_id'],
                'trace_id': str(unit_tags),
                'request_time': start_time,
                'response_time': end_time,
                'duration_time': duration_time,
                'outputs': self.method.result_mark(result)}
            self.method.logger.info({"output": output_log})
            return make_response(jsonify(output), 200)
        else:
            self.method.logger.error(
                {"flask_msg": "result is unexpected!"}, exc_info=True)
            return make_response(jsonify({'error_msg': 'no result!'}), 512)
