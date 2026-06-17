import mlflow
import sys
from loguru import logger


def _get_current_staging_version(client: mlflow.MlflowClient, model_name: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, "staging")
        return mv.version
    except (AttributeError, mlflow.exceptions.MlflowException):
        pass

    try:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            return versions[0].version
    except mlflow.exceptions.MlflowException:
        pass

    return None

def _set_staging_alias(client: mlflow.MlflowClient, model_name: str, version: str):
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=version,
        )
        logger.info(f"Registered {model_name} v{version} → alias 'staging'")
    except AttributeError:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        logger.info(f"Registered {model_name} v{version} → Staging")


def register(run_id: str, model_name: str = "ride_completion") -> str:
    client = mlflow.MlflowClient()

    try:
        result = mlflow.register_model(
            model_uri=f"runs:{run_id}/model",
            name=model_name,
        )
        _set_staging_alias(client, model_name, result.version)
        return result.version

    except Exception as exc:
        logger.error(f"Failed to register model from run {run_id}: {exc}")

        fallback_version = _get_current_staging_version(client, model_name)
        if fallback_version:
            logger.warning(
                f"Falling back to current staging version: "
                f"{model_name} v{fallback_version}"
            )
            return fallback_version

        logger.critical("No existing staging model found — cannot fallback")
        raise


if __name__ == "__main__":
    register(sys.argv[1])
