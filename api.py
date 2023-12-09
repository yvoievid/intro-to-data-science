from flask import Flask, jsonify, request
from operation_simulation.models.inference import Inference
from operation_simulation.helpers.utils import api_response, get_query_params_for_inference, reset_response
import copy


"""
API Documentation 
    
The endpoint /command/simulate accepts command, a flank to choose, a group name, the weather year period, and strategy.

Parameter Format Required Default Description
command String Yes ATTACK Describes the main action that main unit group performs. Possible values: ATTACK
group String Yes ALPHA The name of group that is going to be used for simmulation. After training, the simulation will show how this group performs attack on target, Possible values: ALPHA, DELTA, OMEGA
flank String No CENTER The path that main group try to use to find optimal way to reach the target, Possible values, CENTER, LEFT, RIGHT
weather Strig No SUMMER Season of the year that is used in simulation. SUMMER or WINTER supported. If WINTER choosen it would be considered as difficult terrain conditions that will make agent to stop at some point if the path is too long, Possible values: SUMMBER, WINTER
strategy String NO SAFE The way that main unit group will behave in respect to safetiness. If chossen AGGRESSIVE, than agent will almost ignore the enemies. If SAFE then main unit group will try to ommit the enemies and try to find other safer path, Possible values: SAFE, AGGRESSIVE


The endpoint /command/smoke accepts command, a flank to choose, a group name, the weather year period, and strategy. This endpoint doesn't run the simmulation, it just returns the expected output of /command endpoint

The endpoint /command/reset resets the state of the simulation and brings back all the units to its' initial positions

The endpoint /command/switch-terrain changes the background to the alternative as it is from satilite or compressed in sake of simulation
"""

app = Flask(__name__)

inference = Inference(**{
                'command':"ATTACK",
                'flank': "CENTER",
                'simulate': False,
                'reset': False})


@app.route('/getInference/', methods=['GET'])
def welcome():
    global inference
    original_indefence = copy.deepcopy(inference)
    inference.train = False
    inference.simulate = False
    inference.dryrun = False
    inference.reset = False
    inference.switch_terrain = False
    
    return jsonify(original_indefence)


@app.route('/command/simulate', methods=['GET', 'POST'])
def simulate():
    global inference
    inference = get_query_params_for_inference(request.args)
    inference.simulate = True
    return api_response(inference)

@app.route('/command/dryrun', methods=['GET', 'POST'])
def dryrun():
    global inference
    inference = get_query_params_for_inference(request.args)
    inference.dryrun = True
    return api_response(inference)


@app.route('/command/smoke', methods=['GET', 'POST'])
def smoke():
    global inference
    inference = get_query_params_for_inference(request.args)    
    return api_response(inference)

@app.route('/command/reset', methods=['GET', 'POST'])
def reset():
    global inference
    inference.reset = True
    return reset_response()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
