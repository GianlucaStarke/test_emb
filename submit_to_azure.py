import os
from dotenv import load_dotenv, find_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment, CommandJob
from azure.ai.ml.entities import JobResourceConfiguration

load_dotenv(find_dotenv())

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)

env = Environment(
    name="faiss-env",
    description="Ambiente com faiss e langchain",
    conda_file="env.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

ml_client.environments.create_or_update(env)

job = command(
    code="./", 
    command="python train.py",
    environment="faiss-env",
    compute="cpu-cluster",  # Altere para o nome do seu compute cluster
    experiment_name="treinar-bd-sql",
    display_name="Treinamento com faiss",
)

ml_client.jobs.create_or_update(job)
print("🚀 Job enviado para o Azure!")
