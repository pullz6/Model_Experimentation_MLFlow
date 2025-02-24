import mlflow
from model import * 


def run_deployment():
    print('starting the run')
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")
    
    accuracy,params,X_train, y_train, lr = loading_training()
    print(accuracy)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )
        
if __name__ == "__main__":
    print('starting')
    run_deployment()
    
