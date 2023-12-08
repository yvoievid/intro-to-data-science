from operation_simulation.models.inference import Inference

def api_response(inference):
    return f"""Successfully started simmulation with group {inference.group.upper()} that make {inference.command.upper()}
        through {inference.flank.upper()} flank considering that it is {inference.weather.upper()} year period and using {inference.strategy.upper()} strategy"""

def get_query_params_for_inference(query_params):
    inference = Inference(command=query_params.get("command", default="ATTACK"), 
                        flank=query_params.get("flank", default="CENTER"),
                        weather=query_params.get("weather", default="SUMMER"), 
                        group=query_params.get("group", default="ALPHA"),
                        strategy=query_params.get("strategy",  default="SAFE"))
    
    
    return inference