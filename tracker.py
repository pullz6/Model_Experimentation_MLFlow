import mlflow
from model import * 


def run_deployment():
    print('starting the run')
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("Energy_tracker")
    
    mse,params,X_train, y_train, lr, X_test, y_test = loading_training()
    print(mse)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("MSE", mse)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Training a linear regression model for Engery consumption")

        # Infer the model signature
        signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="Energy_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-model",
        )
        
        return X_test, y_test, model_info

def inference(X_test, y_test, model_info): 
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    result = pd.DataFrame(X_test)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])
    
    
if __name__ == "__main__":
    print('starting')
    X_test, y_test, model_info = run_deployment()
    inference(X_test, y_test, model_info)
    
