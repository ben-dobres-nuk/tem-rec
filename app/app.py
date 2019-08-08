from flask import Flask
from flask import jsonify
import json
import argparse


import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
#context = ('local.crt', 'local.key') #certificate and key files

with open('data/rec_proc.json') as json_file:
    response_data = json.load(json_file)

valid_tags = response_data.keys()

app = Flask(__name__)


@app.route('/')
def index():
    return 'Index Page'


@app.route('/rec/')
def no_tag():
    response = {"message": 'please choose a tag, for example: /rec/[tag]'}
    return jsonify(response)


@app.route('/rec/<tag>')
def reccomend(tag):
    print("tag received: {}".format(tag))

    # return a reccomendation if we have one
    if tag in valid_tags:
        response = {
            "main_tag": tag,
            "rec": response_data[tag],
            "rec_found": True
        }

    # otherwise report that no reccomendation exsits
    else:
        response = {"main_tag": tag, "rec": None, "rec_found": False}

    return jsonify(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port')

    parser.add_argument('--local', dest='local', action='store_true')
    parser.add_argument('--web', dest='local', action='store_false')
    parser.set_defaults(feature=False)
    parser.set_defaults(port=8888)
    args = parser.parse_args()
    port = int(args.port)

    if args.local:
        host = 'localhost'
    else:
        host = '0.0.0.0'

    app.run(port=port, debug=False, host=host, ssl_context='adhoc')
