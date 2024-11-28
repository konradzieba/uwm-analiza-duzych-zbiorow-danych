import textwrap
from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    "tutorial_hourly",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="An hourly tutorial DAG with updated documentation and tags",
    schedule=timedelta(hours=1),
    start_date=datetime.now(),
    catchup=False,
    tags=["lab8"],
) as dag:

    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )

    t2 = BashOperator(
        task_id="sleep",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=3,
    )
    t1.doc_md = textwrap.dedent(
        """\
    #### Task Documentation
    This task prints the current date and time.
    """
    )

    dag.doc_md = """
    ### DAG Documentation
    This DAG runs hourly to demonstrate the use of BashOperators in Airflow.
    It contains three tasks:
    1. Print the current date and time.
    2. Sleep for 5 seconds.
    3. Print templated output using Jinja2.
    """
    templated_command = textwrap.dedent(
        """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7)}}"
    {% endfor %}
    """
    )

    t3 = BashOperator(
        task_id="templated",
        depends_on_past=False,
        bash_command=templated_command,
    )

    t1 >> [t2, t3]
