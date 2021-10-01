"""Example workflow pipeline script for customer churn pipeline.
                                                                                                         . BatchTransform
                                                                       . -RegisterModel -> CreateModel  . 
                                                                     .                                   . SageMakerClarify
    Process-> HyperParameterTuning -> EvaluateBestModel -> Condition .
                                                                      . -(stop)
Implements a get_pipeline(**kwargs) method.
"""
import os
import json
import boto3
import sagemaker
import sagemaker.session
import datetime as dt
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model import Model
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.inputs import CreateModelInput
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.transformer import Transformer
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tuner import HyperparameterTuner
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TuningStep
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="ChurnModelPackageGroup",
    pipeline_name="ChurnModelPipeline",
    base_prefix = None,
    custom_image_uri = None,
    sklearn_processor_version=None
    ):
    """Gets a SageMaker ML Pipeline instance working with churn data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount",
    default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.xlarge"
    )
    input_data = ParameterString(
        name="InputData",
        default_value="s3://{}/data/storedata_total.csv".format(default_bucket),
    )
    batch_data = ParameterString(
        name="BatchData",
         default_value="s3://{}/data/batch/batch.csv".format(default_bucket),
    )
    
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_processor_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="ChurnModelProcess",
        processor=sklearn_processor,
        inputs=[
          ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),  
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train",\
                             destination=f"s3://{default_bucket}/output/train" ),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation",\
                            destination=f"s3://{default_bucket}/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test",\
                            destination=f"s3://{default_bucket}/output/test")
        ],
        code=f"s3://{default_bucket}/input/code/preprocess.py",
    )
    
    # training step for generating model artifacts
    model_path = f"s3://{default_bucket}/output"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    fixed_hyperparameters = {
    "eval_metric":"auc",
    "objective":"binary:logistic",
    "num_round":"100",
    "rate_drop":"0.3",
    "tweedie_variance_power":"1.4"
    }
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        hyperparameters=fixed_hyperparameters,
        output_path=model_path,
        base_job_name=f"churn-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    hyperparameter_ranges = {
    "eta": ContinuousParameter(0, 1),
    "min_child_weight": ContinuousParameter(1, 10),
    "alpha": ContinuousParameter(0, 2),
    "max_depth": IntegerParameter(1, 10),
    }
    objective_metric_name = "validation:auc"
    
    step_tuning = TuningStep(
    name = "ChurnHyperParameterTuning",
    tuner = HyperparameterTuner(xgb_train, objective_metric_name, hyperparameter_ranges, max_jobs=2, max_parallel_jobs=2),
    inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-churn-eval",
        role=role,
        sagemaker_session=sagemaker_session,
    )
    evaluation_report = PropertyFile(
        name="ChurnEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="ChurnEvalBestModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",\
                             destination=f"s3://{default_bucket}/output/evaluation"),
        ],
        code=f"s3://{default_bucket}/input/code/evaluate.py",
        property_files=[evaluation_report],
    )
    
    # step to create model 
    model = Model(
        image_uri=image_uri,        
        model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
        sagemaker_session=sagemaker_session,
        role=role,
    )
    inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )
    step_create_model = CreateModelStep(
        name="ChurnCreateModel",
        model=model,
        inputs=inputs,
    )
    
    # step to perform batch transformation
    transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=f"s3://{default_bucket}/ChurnTransform"
    )
    step_transform = TransformStep(
    name="ChurnTransform",
    transformer=transformer,
    inputs=TransformInput(data=batch_data,content_type="text/csv")
    )
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="s3://{}/evaluation.json".format(default_bucket),
        content_type="application/json"
        )
    )
    step_register = RegisterModel(
        name="RegisterChurnModel",
        estimator=xgb_train,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=default_bucket,prefix="output"),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
    )
    
    # Processing Step to generate analysis config file for Clarify Step
    bias_report_output_path = f"s3://{default_bucket}/clarify-output/bias"
    clarify_instance_type = 'ml.c5.xlarge'
    analysis_config_path = f"s3://{default_bucket}/clarify-output/bias/analysis_config.json"
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri=custom_image_uri,
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
    )
    step_config_file = ProcessingStep(
        name="ChurnModelConfigFile",
        processor=script_processor,
        code=f"s3://{default_bucket}/input/code/generate_config.py",
        job_arguments=["--modelname",step_create_model.properties.ModelName,"--bias-report-output-path",bias_report_output_path,"--clarify-instance-type",clarify_instance_type,\
                      "--default-bucket",default_bucket,"--num-baseline-samples","50","--instance-count","1"],
        depends_on= [step_create_model.name]
    )
    
    # clarify step
    data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=f's3://{default_bucket}/output/train/train.csv',
    s3_output_path=bias_report_output_path,
        label=0,
        headers= ['target','esent','eopenrate','eclickrate','avgorder','ordfreq','paperless','refill','doorstep','first_last_days_diff','created_first_days_diff','favday_Friday','favday_Monday','favday_Saturday','favday_Sunday','favday_Thursday','favday_Tuesday','favday_Wednesday','city_BLR','city_BOM','city_DEL','city_MAA'],
        dataset_type="text/csv",
    )
    clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type=clarify_instance_type,
        sagemaker_session=sagemaker_session,
    )
    config_input = ProcessingInput(
        input_name="analysis_config",
        source=analysis_config_path,
        destination="/opt/ml/processing/input/analysis_config",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_compression_type="None",
        )
    data_input = ProcessingInput(
        input_name="dataset",
        source=data_config.s3_data_input_path,
        destination="/opt/ml/processing/input/data",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type=data_config.s3_data_distribution_type,
        s3_compression_type=data_config.s3_compression_type,
    )
    result_output = ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=data_config.s3_output_path,
        output_name="analysis_result",
        s3_upload_mode="EndOfJob",
    )
    step_clarify = ProcessingStep(
        name="ClarifyProcessingStep",
        processor=clarify_processor,
        inputs= [data_input, config_input],
        outputs=[result_output],
        depends_on = [step_config_file.name]
    )
    
    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="classification_metrics.auc_score.value"
        ),
        right=0.75,
    )
    step_cond = ConditionStep(
        name="CheckAUCScoreChurnEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register, step_create_model, step_config_file,step_transform,step_clarify],
        else_steps=[],
    )
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data,
            batch_data,
        ],
        steps=[step_process,step_tuning,step_eval,step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline