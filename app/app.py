from flask import Flask
from flask import jsonify
import json

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
    app.run(port=8888, debug=True, host='0.0.0.0')
