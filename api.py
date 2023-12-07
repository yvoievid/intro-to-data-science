from flask import Flask, jsonify, request
from operation_simulation.models.inference import Inference
import copy

app = Flask(__name__)

inference = Inference(**{
                'command':"ATTACK",
                'flang': "CENTER",
                'simulate': False,
                'train': False,
                'dryrun': False})


@app.route('/getInference/', methods=['GET'])
def welcome():
    global inference
    original_indefence = copy.deepcopy(inference)
    inference.train = False
    inference.simulate = False
    inference.dryrun = False
    return jsonify(original_indefence)


@app.route('/command/train', methods=['GET', 'POST'])
def train():
    global inference
    inference = Inference(**request.json)
    inference.train = True
    return jsonify(inference)


@app.route('/command/simulate', methods=['GET', 'POST'])
def simulate():
    global inference
    inference = Inference(**request.json)
    inference.simulate = True
    return jsonify(inference)


@app.route('/command/dryrun', methods=['GET', 'POST'])
def dryrun():
    global inference
    inference = Inference(**request.json)
    inference.dryrun = True
    return jsonify(inference)


@app.route('/command/smoke', methods=['GET', 'POST'])
def smoke():
    global inference
    inference = Inference(**request.json)
    return f"""Successfully started simmulation with group {inference.group} that make {inference.command}
        through {inference.flang} flang considering that it is {inference.weather} year period and using {inference.strategy} strategy"""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
