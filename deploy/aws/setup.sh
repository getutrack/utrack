#!/bin/bash
# Author: Ismail Dawoodjee
# Bash script to automate setup for an EC2 instance using the AWS CLI commands.
# Can add more variables and options as necessary for the commands.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo 'ERROR: Unspecified profile name.' \
        'You must enter your profile name as "./setup.sh profile-name".'
    exit 0
fi

# Ensure that docker is running
if [ "$(systemctl is-active docker)" != "active" ]; then
    echo 'Error: Docker is not running.' \
        'Start Docker using "sudo service docker start".'
    exit 1
fi

# reference: Stackoverflow-Redirect echo output in shell script to logfile
LOG_LOCATION="./logs"
exec > >(tee -a $LOG_LOCATION/setup.log)
exec 2>&1

START_TIME=$(date +"%s")
DATE_TIME=$(date +"%x %r %S %Z")
echo -e "INFO: Started setup at $DATE_TIME \n"

USER_NAME="ismaildawoodjee"
REPO_NAME="flask-app"
POLICY_NAME="AmazonEC2FullAccess"
POLICY_ARN="arn:aws:iam::aws:policy/$POLICY_NAME"

KEYPAIR_NAME="docker-ec2"
SG_NAME="docker-ec2-sg"
IP_ADDRESS="0.0.0.0/0"  # set 0.0.0.0/0 for public to view the hosted app
EC2_TYPE="t2.micro"
IMAGE_ID="ami-018c1c51c7a13e363"  # image for virtual machine, not a Docker image
TAG_KEY="docker"
TAG_VALUE="ec2"

function dockerize_and_push () {
    # A Docker repository is required for this step
    echo -e "INFO: Building Docker image and pushing to Docker Hub \n"
    sudo docker build -t "$USER_NAME/$REPO_NAME:ec2" .
    sudo docker push "$USER_NAME/$REPO_NAME:ec2"
}

function attach_policy () {
    # Assign IAM permission for user to have full EC2 access
    echo -e "INFO: Attaching user policy $POLICY_NAME to $1 \n"
    aws iam attach-user-policy \
        --policy-arn $POLICY_ARN \
        --user-name "$1"
}

function create_authorize_security_group () {
    echo -e "INFO: Creating security group $SG_NAME \n"
    aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group to host Docker container"

    # Authorize security group to allow SSH access into EC2
    echo -e "INFO: Authorizing security group $SG_NAME for SSH access \n"
    aws ec2 authorize-security-group-ingress \
        --group-name $SG_NAME \
        --protocol tcp \
        --port 22 \
        --cidr $IP_ADDRESS

    # To allow Flask app (port 5000) to be viewable from anywhere
    aws ec2 authorize-security-group-ingress \
        --group-name $SG_NAME \
        --protocol tcp \
        --port 5000 \
        --cidr $IP_ADDRESS
}

function create_keypair () {
    # Create a key pair for the securely launching EC2 instance
    echo -e "INFO: Creating a key pair with name $KEYPAIR_NAME \n"
    aws ec2 create-key-pair \
        --key-name $KEYPAIR_NAME \
        --query "KeyMaterial" \
        --output text \
        > $KEYPAIR_NAME.pem

    # Change permissions to read-only for user, others should not have access
    chmod 400 $KEYPAIR_NAME.pem
}

function launch_ec2 () {
    echo -e "INFO: Launching an instance of type $EC2_TYPE in security group $SG_NAME \n"
    aws ec2 run-instances \
        --image-id $IMAGE_ID \
        --count 1 \
        --instance-type $EC2_TYPE \
        --key-name $KEYPAIR_NAME \
        --security-groups $SG_NAME \
        --tag-specifications \
            "ResourceType=instance,Tags=[{Key=$TAG_KEY,Value=$TAG_VALUE}]"
}

function access_with_ssh () {
    # Assuming there is ONLY one instance with a unique key-value for tag
    echo -e "INFO: Getting public IP address for EC2 instance \n"
    EC2_PUBLIC_DNS=$(aws ec2 describe-instances \
        --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" \
        --query "Reservations[].Instances[].PublicDnsName" \
        --output text
    )

    # Wait 30 seconds to allow EC2 instance to fully boot up
    echo -e "INFO: Waiting 30 seconds. Copying deployment script and accessing with SSH \n"
    sleep 30
    scp -i "$KEYPAIR_NAME.pem" deploy.sh "ec2-user@$EC2_PUBLIC_DNS:/home/ec2-user"

    # Retry until SSH access is possible
    until ssh -i "$KEYPAIR_NAME.pem" "ec2-user@$EC2_PUBLIC_DNS"; do
        sleep 5
    done
}

dockerize_and_push
attach_policy "$@"
create_keypair
create_authorize_security_group
launch_ec2
access_with_ssh

END_TIME=$(date +"%s")
DURATION=$((END_TIME - START_TIME))
echo "INFO: Completed setup within $DURATION seconds."
