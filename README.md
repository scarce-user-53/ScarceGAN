## Introduction

This is the Code submission accompanying the CIKM 2021 paper, titled "ScarceGAN: Discriminative Classification Framework for Rare
Class Identification for Longitudinal Data with Weak Prior"

This code contains the implementations of baseline Vanilla SSGAN, AAE, ScarceGAN, ScarceGAN w/o bad generator, and ScarceGAN w/o leeway term employed in the paper to generate the desired results. Entire code is written using Keras with TensorFlow2.x as backend. In its current state, the code can be used to train new models, while pre-trained models will be made available shortly.

We specifically address: 

1) Severe scarcity in positive class, stemming from both underlying organic skew in the data, as well as extremely limited labels

2) Multi-class nature of the negative samples, with uneven density distributions and partially overlapping feature distributions

3) Massively unlabeled data leading to tiny and weak prior on both positive and negative classes, and possibility of unseen or unknown behavior in the unlabelled set, especially in the negative class

ScarceGAN re-formulates semi-supervised GAN by accommodating weakly labelled multi-class negative samples and the available positive samples. It relaxes
the supervised discriminator’s constraint on exact differentiation between negative samples by introducing a ‘leeway’ term for samples with noisy prior.
</p>
Below figure shows the architecture of ScarceGAN
</p>
<div>
<img src="https://github.com/scarce-user-53/ScarceGAN/blob/master/images/ScarceGAN.png" alt="ScarceGAN" width="400" height="420" >
</div>

## Dependencies and Environment
Dependencies can be installed via anaconda. Below are the dependencies:
```
Keras with TensorFlow2.0 as backend environment:
- pip=20.0.2
- python=3.6.10
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - tensorflow-addons
    - tensorflow-estimator==2.2.0
```
```
For running the feature extraction jobs we used Amazon Elastic Map Reduce Framework with r4.4xlarge fleet instances:
- pip=20.0.2
- spark=3.0.1
- Hadoop=3.2.1
- Hive=3.1.2
- pystan=2.17
- fbprophet=0.7.1
- python=3.6.10
- EMR:
    - executor-cores 16
    - executor-memory 20g 
    - num-executors 20 
    - driver-memory 55g
    - driver-cores 32
- pip:
    - pandas==1.1.5
    - numpy==1.18.1
    - boto3==1.17.76
```

	
    
## Training Data
Data for training can be collected by executing ``feature_extraction/Extract_Prophet_Features.py`` in a spark environment

	
## Training Steps
The code provides training procedure for baseline AAE, Vanilla SSGAN, ScarceGAN, ScarceGAN w/o bad generator, and ScarceGAN w/o leeway term. Additionally, implementation of ScarceGAN for KDDCup99 dataset has been included in the implementation for comparison.   


1) The fastest way to train all the variants of SSGAN model is by running the py file in ``inference/SSGAN_Inference.py``. To get the results on test case for other SSGAN variants, pass the trained model to ``inference/Explainability.py``

2) Aternatively, you can train any model of your choice by running ``training/SSGAN/<gan_file>.py``

## Reference
To run the spark jobs, run the below command in the master node:
```
time spark-submit \
--conf spark.kubernetes.namespace=admin \
--conf spark.driver.maxResultSize=18g \
--conf spark.default.parallelism=320 \
--conf spark.sql.shuffle.partitions=4800 \
--conf spark.memory.fraction=1 \
--conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
--conf spark.memory.storageFraction=0.9 \
--conf spark.executor.memoryOverhead=10g \
--conf spark.driver.memoryOverhead=5g \
--conf spark.python.worker.memory=6g \
--conf spark.python.worker.reuse=false \
--conf spark.task.cpus=1 \
--conf spark.shuffle.useOldFetchProtocol=true \
--conf spark.task.maxFailures=100 \
--conf spark.metrics.conf.*.sink.graphite.class=org.apache.spark.metrics.sink.GraphiteSink \
--conf spark.metrics.conf.*.sink.graphite.host=influxdb.monitoring.svc.cluster.local \
--conf spark.metrics.conf.*.sink.graphite.port=2003 \
--conf spark.metrics.conf.*.sink.graphite.period=2 \
--conf spark.metrics.conf.*.sink.graphite.unit=seconds \
--conf spark.metrics.conf..source.jvm.class=org.apache.spark.metrics.source.JvmSource \
--conf spark.eventLog.logStageExecutorMetrics=true \
--conf spark.ui.prometheus.enabled=true \
--conf spark.metrics.appStatusSource.enabled=true \
--conf spark.sql.streaming.metricsEnabled=true \
--conf spark.executor.processTreeMetrics.enabled=true \
--conf spark.metrics.conf.*.sink.graphite.prefix=kubeflow \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=default-editor \
--master k8s://https://kubernetes.default.svc:443 \
--deploy-mode cluster \
--executor-cores 16 \
--executor-memory 20g \
--num-executors 20 \
--driver-memory 55g \
--driver-cores 32 \
--py-files s3a://bucket_name/pyfiles/Raw_Data_Collection.py\
s3a://bucket_name/pyfiles/Raw_Data_Collection.py
```
