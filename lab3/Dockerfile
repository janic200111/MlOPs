# Use the official AWS Lambda Python 3.10 image as the base image
FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements file into the container
COPY requirements.txt ./ 

# Install the dependencies
RUN pip install -r requirements.txt

# Copy lambda function code into the container
COPY ./lambda_src/ ${LAMBDA_TASK_ROOT}

# Copy the model
COPY ./model/ ${LAMBDA_TASK_ROOT}/model

# Run lambda
CMD ["lambda.lambda_handler"]