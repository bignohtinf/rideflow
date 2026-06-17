import great_expectations as gx 
import pandas as pd 
import sys 

def run_checkpoint(parquet_path: str, suite_name: str) -> bool:
    context = gx.get_context()

    df = pd.read_parquet(parquet_path)
    datasource = context.sources.add_or_update_pandas(name="spark_output")
    asset = datasource.add_dataframe_asset(name="batch")
    batch = asset.build_batch_request(dataframe=df)

    results = context.run_checkpoint(
        checkpoint_name="rides_checkpoint",
        validations=[{
            "batch_request": batch,
            "expectation_suite_name": suite_name
        }]
    )

    if not results.success:
        raise ValueError(f"Validation failed: {results}")
    
    return True 

if __name__ == "__main__":
    parquet_path = sys.argv[1]
    suite_name = sys.argv[2]

    run_checkpoint(parquet_path, suite_name)