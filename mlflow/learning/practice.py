from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score


import mlflow


mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("Breast cancer classification")
mlflow.sklearn.autolog()


X, y = load_breast_cancer(return_X_y=True)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    n_estimators = 10
    max_depth = 3
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(x_test)

    # acc = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)


    # mlflow.log_params(
    #     {
    #         'n_estimators' : n_estimators,
    #         'max_depth' : max_depth
    #     }
    # )

    # mlflow.log_metrics({
    #     'accuracy' : acc,
    #     'f1-score' : f1
    # })

    # signature = mlflow.models.infer_signature(X_train, model.predict(x_test))
    # mlflow.sklearn.log_model(
    #     model,
    #     name='RandomForest-baseline',
    #     signature=signature,
    #     input_example=X_train[:10]
    # )

