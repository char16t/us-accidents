import bentoml

runner = bentoml.mlflow.get("mymodel:latest").to_runner()

svc = bentoml.Service('mymodel', runners=[runner])


@svc.api(input=bentoml.io.NumpyNdarray(), output=bentoml.io.NumpyNdarray())
def predict(input_data: str):
    return runner.predict.run([input_data])[0]
