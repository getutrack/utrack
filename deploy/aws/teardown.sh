#!/bin/bash
# Author: Ismail Dawoodjee
# This script tears down the infrastructure built from setup.sh. It:
# - removes key-pair on local host
# - removes key-pair stored on AWS
# - wipes the log files (optional)
# - terminates the EC2 instance that was spun up
# - deletes the security group created for the EC2 instance

LOG_LOCATION="./logs"
exec > >(tee -a $LOG_LOCATION/teardown.log)
exec 2>&1

KEYPAIR_NAME="docker-ec2"
SG_NAME="docker-ec2-sg"
TAG_KEY="docker"
TAG_VALUE="ec2"

# > ./logs/setup.log
# rm logs/*

echo -e "Tearing down local and cloud infrastructure \n"
rm -f "$KEYPAIR_NAME.pem"

aws ec2 delete-key-pair --key-name $KEYPAIR_NAME

INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text
)

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"

sleep 60
aws ec2 delete-security-group --group-name $SG_NAME