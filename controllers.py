from flask import Flask, jsonify, request
from operation_simulation.models.inference import Inference
import copy

app = Flask(__name__)

inference = Inference(**{
                'strategy':"DEFEND",
                'flang': "CENTER",
                'simulate': False,
                'train': False })


@app.route('/getInference/', methods=['GET'])
def welcome():
    global inference
    original_indefence = copy.deepcopy(inference)
    inference.train = False
    inference.simulate = False
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

@app.route('/command/reset', methods=['GET', 'POST'])
def reset_inference():
    global inference
    inference.train = False
    inference.simulate = False
    return "SUCCESS"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
