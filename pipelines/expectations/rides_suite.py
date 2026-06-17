import json
import great_expectations as gx 
from pathlib import Path 

def build_suite(json_path: str) -> gx.core.ExpectationSuite:
    context = gx.get_context()
    with open(json_path) as f:
        config = json.load(f)
    
    suite = context.add_or_update_expectation_suite(
        expectation_suite_name=config["suite_name"]
    )

    for exp in config["expectations"]:
        exp_type = exp.pop("type")
        suite.add_expectation(
            gx.core.ExpectationConfiguration(
                expectation_type=exp_type,
                kwargs=exp
            )
        )
    context.save_expectation_suite(suite)
    return suite