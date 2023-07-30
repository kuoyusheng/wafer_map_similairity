import torch
from sagemaker.pytorch import PyTorch
import boto3
from sagemaker import TrainingInput

iam = boto3.client('iam')
role = iam.get_role(RoleName='dev')['Role']['Arn']
region = boto3.Session().region_name
print(region)
#`print(help(PyTorch))
if __name__ == '__main__':
    bucket = 'yusheng-aws'
    key = 'test-model'
    s3_output_location = f's3://{bucket}/{key}/output'
    input_train = TrainingInput(s3_data='s3://yusheng-aws/MNIST')
    data_channel = dict(train = input_train) 
    # training script
    pytorch_estimator = PyTorch(entry_point='train.py', 
                                role=role,
                                instance_type= 'ml.m4.xlarge', 
                                instance_count = 1,
                                framework_version='2.0.0',
                                py_version='py310',
                                output_path = s3_output_location,
                                hyperparameters = {'epochs': 3, 'batch-size': 64, 'learning-rate': 1e-3}
                                )
    pytorch_estimator.fit(inputs=data_channel, logs = True)
