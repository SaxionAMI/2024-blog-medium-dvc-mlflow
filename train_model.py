from sklearn.svm import SVC
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_experiment("SVC-model-training") # Will create and set new experiment
# experiment = mlflow.get_experiment_by_name("experiment name")
# enable autologging
mlflow.sklearn.autolog()

# Load dataset
print('Loading train dataset')
train = pd.read_csv('data/train.csv')
X_train = train.drop('label', axis=1)
y_train = train.label

# Train model
with mlflow.start_run() as run:
    print('Training model')
    clf = SVC(kernel='rbf', probability=True) #If you used additional parameters then you should also save them in a file
    clf.fit(X_train, y_train)
    print('Saving model')
    joblib.dump(clf, f"models/SVC/model")
    print('Saving parameters')
    # Save the variables we used
    run_id = run.info.run_id
    run_name = mlflow.get_run(run_id).data.tags["mlflow.runName"]
    parameters = {'run_id': run_id, 'run_name': run_name}
    with open("models/SVC/model_parameters.yaml", "w") as f: f.write('\n'.join(f"{k}: {v}" for k, v in parameters.items()))
print('Done')