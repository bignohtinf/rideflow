from dagster import job, op
from models.training.train import train
from models.training.evaluate import is_better_than_production
from models.training.register_model import register

@op(config_schema={"target_date": str, "model_name": str})
def run_train(context):
    target_date = context.op_config["target_date"]
    model_name  = context.op_config["model_name"]
    run_id, metrics = train(target_date, model_name)
    context.add_output_metadata({"run_id": run_id, "auc": metrics["AUC-ROC"]})
    return run_id, metrics

@op
def run_register_if_better(context, result):
    run_id, metrics = result
    if is_better_than_production(metrics):
        version = register(run_id)
        context.log.info(f"Registered v{version} → Staging")
    else:
        context.log.info("New model không tốt hơn production, bỏ qua")

@job
def retrain_pipeline():
    run_register_if_better(run_train())