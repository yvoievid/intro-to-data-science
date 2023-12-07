
def api_response(inference):
    return f"""Successfully started simmulation with group {inference.group.upper()} that make {inference.command.upper()}
        through {inference.flang.upper()} flang considering that it is {inference.weather.upper()} year period and using {inference.strategy.upper()} strategy"""

