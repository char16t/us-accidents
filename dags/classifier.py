import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import bentoml
import mlflow
import git
import os
import sys
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, get_current_context
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from os.path import dirname, abspath

my_dir_path = dirname(dirname(abspath(__file__)))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['valeriy@manenkov.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}


def build_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("My experiment")
    mlflow.autolog()

    repo = git.Repo(search_parent_directories=True)
    sha_commit = repo.head.object.hexsha
    git_branch = repo.head.name

    run = mlflow.start_run()
    mlflow.set_tag("mlflow.runName", datetime.now().strftime("MyExperiment-%Y-%m-%d-%H-%M-%S"))
    mlflow.set_tag("mlflow.note.content", "My experiment with XGBClassifier")
    mlflow.set_tag("mlflow.source.type", "JOB")
    mlflow.set_tag("mlflow.source.name", "classifier.py")
    mlflow.set_tag("mlflow.source.git.commit", sha_commit)
    mlflow.set_tag("mlflow.source.git.branch", git_branch)
    mlflow.set_tag("mlflow.source.git.repoURL", "https://github.com/char16t/us-accidents")
    mlflow.set_tag("mlflow.project.env", "poetry")

    df = pd.read_csv('data/US_Accidents_Dec20_updated.csv')
    df = df.drop(['ID', 'Country', 'Turning_Loop'], axis=1)
    df = df.drop(['Distance(mi)', 'End_Time', 'End_Lat', 'End_Lng', 'Description'], axis=1)
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())
    df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(df['Wind_Chill(F)'].median())
    df = df.dropna(subset=['City', 'Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])
    df = df.dropna(subset=['Zipcode'])
    df = df.dropna(subset=['Timezone'])
    df = df.dropna(subset=['Airport_Code'])
    df = df.dropna(subset=['Weather_Timestamp'])
    df = df.dropna(subset=['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)'])
    df = df.dropna(subset=['Weather_Condition'])
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'])
    mapping = {
        "N": ["NNW", "NW", "North", "NNE", "NE", "N"],
        "S": ["SW", "South", "SSE", "SSW", "SE", "S"],
        "W": ["WSW", "WNW", "West", "W"],
        "E": ["ESE", "East", "ENE", "E"],
        "VAR": ["Variable", "VAR"],
        "CALM": ["CALM"]
    }

    for newv, oldvs in mapping.items():
        df.loc[df['Wind_Direction'].isin(oldvs),'Wind_Direction'] = newv     

    df['Clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na = False), True, False)
    df['Cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), True, False)
    df['Rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), True, False)
    df['Heavy_Rain'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), True, False)
    df['Snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), True, False)
    df['Heavy_Snow'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), True, False)
    df['Fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na = False), True, False)
    df = df.drop(['Weather_Condition'], axis=1)

    df['Side'] = df['Side'].astype('category')
    df['City'] = df['City'].astype('category')
    df['County'] = df['County'].astype('category')
    df['State'] = df['State'].astype('category')
    df['Airport_Code'] = df['Airport_Code'].astype('category')
    df['Timezone'] = df['Timezone'].astype('category')
    df['Wind_Direction'] = df['Wind_Direction'].astype('category')
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].astype('category')
    df['Civil_Twilight'] = df['Civil_Twilight'].astype('category')
    df['Nautical_Twilight'] = df['Nautical_Twilight'].astype('category')
    df['Astronomical_Twilight'] = df['Astronomical_Twilight'].astype('category')

    df['Street'] = df['Street'].astype('string')
    df['Zipcode'] = df['Zipcode'].astype('string')

    fre_list = ['Street', 'City', 'County', 'Zipcode', 'Airport_Code','State']
    for i in fre_list:
        newname = i + '_Freq'
        df[newname] = df.groupby([i])[i].transform('count')
        df[newname] = df[newname]/df.shape[0]*df[i].unique().size
    df = df.drop(fre_list, axis  = 1)

    checks = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    for c in checks:
        Q1, Q3 = df[c].quantile([.25, .75])
        IQR = Q3 - Q1
        values_lower = Q1 - 1.5*IQR
        values_upper = Q3 + 1.5*IQR
        rows_to_drop = df[
            (df[c] < values_lower) | (df[c] > values_upper)].index
        df = df.drop(rows_to_drop)

    cat_features = ['Side','Timezone','Wind_Direction', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

    enc = OneHotEncoder(drop='if_binary', sparse=False)
    enc.fit(df[cat_features])
    dummies = pd.DataFrame(enc.transform(df[cat_features]), 
                        columns=enc.get_feature_names_out(), index=df.index)
    df = pd.concat((df, dummies), axis=1).drop(cat_features, axis=1)
    df = df.replace([True, False], [1,0])

    seconds_in_day = 24*60*60

    df['Start_Time_sin'] = df['Start_Time'].apply(lambda x: np.sin(2*np.pi*datetime.timestamp(x)/seconds_in_day))
    df['Start_Time_cos'] = df['Start_Time'].apply(lambda x: np.cos(2*np.pi*datetime.timestamp(x)/seconds_in_day))
    df = df.drop(["Start_Time"], axis=1)
    df['Weather_Timestamp_sin'] = df['Weather_Timestamp'].apply(lambda x: np.sin(2*np.pi*datetime.timestamp(x)/seconds_in_day))
    df['Weather_Timestamp_cos'] = df['Weather_Timestamp'].apply(lambda x: np.cos(2*np.pi*datetime.timestamp(x)/seconds_in_day))
    df = df.drop(["Weather_Timestamp"], axis=1)

    df = df.loc[:,~df.columns.duplicated()]

    X = df.drop(['Severity'], axis=1)
    y = df[['Severity']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    xgb_clf = xgb.XGBClassifier(objective='multi:softmax', eval_metric=['merror','mlogloss'], seed=42)
    xgb_clf.fit(X_train, 
                y_train,
                verbose=0,
                eval_set=[(X_train, y_train), (X_test, y_test)])

    extra_pip_requirements=["xgboost"]
    mlflow.xgboost.log_model(
        xgb_clf, "model", extra_pip_requirements=extra_pip_requirements)
    registered_model = mlflow.register_model(
        "runs:/{}/model".format(run.info.run_id), "mymodel")
    bentoml.mlflow.import_model(
            "mymodel",
            registered_model.source,
            signatures={"predict": {"batchable": True}},
        )


    mlflow.end_run()

    client = MlflowClient()
    client.transition_model_version_stage(
        name="mymodel",
        version=registered_model.version,
        stage="Production",
    )
    context = get_current_context()
    context["ti"].xcom_push(key='model_uri', value=registered_model.source.replace("mlflow-artifacts:", my_dir_path + "/mlartifacts", 1))

dag = DAG('mymodel', default_args=default_args, schedule_interval=timedelta(days=1))

# Версионирование данных в DVC.
# Предполагается, что последние версии данных в .csv уже загружены.
# Теоретически, данные могли измениться с последнего запуска.
tt1 = BashOperator(
    task_id="version_data",
    bash_command="cd {} && dvc add data/iris.csv && dvc push".format(
        my_dir_path),
    dag=dag
)

# Версионирование кода в Git
tt2 = BashOperator(
    task_id="stage_files",
    bash_command="cd {} && git add .".format(my_dir_path),
    dag=dag
)

tt3 = BashOperator(
    task_id="commit_files",
    bash_command="cd {} && git commit -m 'Update data file' || echo 'No changes to commit'".format(
        my_dir_path),
    dag=dag
)

tt4 = PythonOperator(
    task_id="train_model",
    python_callable=build_model,
    provide_context=True,
    dag=dag
)

bento_path = os.path.join(my_dir_path, "include/bentoml")

# We export BENTOML_MLFLOW_MODEL_PATH to be used in the bentofile.yaml so BentoML can find the model's requirements.txt
tt5 = BashOperator(
    task_id="build_model",
    bash_command="cd {} && export BENTOML_MLFLOW_MODEL_PATH={{{{ ti.xcom_pull(key='model_uri') }}}} && bentoml build".format(
        bento_path),
    dag=dag
)

tt6 = BashOperator(
    task_id="containerize_model",
    bash_command="bentoml containerize mymodel:latest -t mymodel:latest",
    dag=dag
)

docker_compose_file_path = os.path.join(
    my_dir_path, "include/docker-compose/docker-compose.yml")


tt7 = BashOperator(
    task_id="serve_model",
    bash_command="docker compose -f {} up -d --wait".format(
        docker_compose_file_path),
    dag=dag
)

tt1 >> tt2 >> tt3 >> tt4 >> tt5 >> tt6 >> tt7
