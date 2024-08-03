FROM amazonlinux:2

# Set environment variables
ENV SPARK_VERSION=2.4.7
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:/opt/spark/bin
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Install necessary packages
RUN yum update -y && \
    yum install -y \
    java-1.8.0-openjdk-devel \
    python3 \
    python3-pip \
    wget \
    tar \
    gcc \
    python3-devel && \
    yum clean all

# Verify Python and pip installation
RUN python3 --version && pip3 --version

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy==1.21.6 \
    pandas==1.3.5 \
    pyspark==2.4.7

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar xvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz

# Copy the prediction script to the /app directory
COPY TestSimple.py /app/TestSimple.py
COPY ValidationDataset.csv /app/ValidationDataset.csv

# Copy pre-trained models
COPY random_forest_model /app/random_forest_model
COPY logistic_regression_model /app/logistic_regression_model
COPY decision_tree_model /app/decision_tree_model

# Set the working directory to /app
WORKDIR /app

# Set the entrypoint to run the validation script
ENTRYPOINT ["python3", "TestSimple.py"]
