#!/bin/bash

# Variables
AWS_REGION="eu-west-1"
INSTANCE_TYPE="t2.micro"
KEY_NAME="ledger-key-pair"
SECURITY_GROUP="utrack-app-sg"
AMI_ID="ami-047bb4163c506cd98" # Ensure this is Amazon Linux 2
TAG_KEY="app"
TAG_VALUE="utrack"
DOCKER_COMPOSE_VERSION="1.29.2"
VOLUME_SIZE=20 # Increase this value as needed
DOMAIN_NAME="getledger.com"
SUBDOMAIN="app"
NETLIFY_TARGET="your-site.netlify.app"  # Replace with your actual Netlify domain
MINTLIFY_TARGET="your-docs.mintlify.app"  # Replace with your actual Mintlify domain

# Create a key pair
if [ ! -f "$KEY_NAME.pem" ]; then
  echo "Creating a new key pair: $KEY_NAME"
  aws ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text --region $AWS_REGION > $KEY_NAME.pem
  chmod 400 $KEY_NAME.pem
else
  echo "Key pair $KEY_NAME already exists."
fi

# Create a security group
aws ec2 create-security-group --group-name $SECURITY_GROUP --description "Security group for Ledger app EC2 instance" --region $AWS_REGION

# Add rules to the security group
aws ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $AWS_REGION
aws ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $AWS_REGION

# Launch an EC2 instance with increased storage
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $KEY_NAME --security-groups $SECURITY_GROUP --block-device-mappings DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_SIZE} --tag-specifications "ResourceType=instance,Tags=[{Key=$TAG_KEY,Value=$TAG_VALUE}]" --query "Instances[0].InstanceId" --output text --region $AWS_REGION)

echo "Waiting for instance to be in running state..."
while state=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].State.Name" --output text --region $AWS_REGION); test "$state" = "pending"; do
  echo -n "."
  sleep 5
done

echo "Instance state: $state"

# Wait for the instance status checks to pass
echo "Waiting for instance status checks to pass..."
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID --region $AWS_REGION

# Get the public DNS of the instance
INSTANCE_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region $AWS_REGION)

echo "Instance is running at $INSTANCE_PUBLIC_IP"

# Add a delay to allow SSH service to start
echo "Waiting for SSH service to be ready..."
sleep 30

# SSH into the instance and set up Docker, Docker Compose, and basic security
ssh -o "StrictHostKeyChecking=no" -i "$KEY_NAME.pem" ec2-user@$INSTANCE_PUBLIC_IP << EOF
  sudo yum update -y
  sudo yum install -y docker
  sudo service docker start
  sudo usermod -a -G docker ec2-user
  sudo curl -L "https://github.com/docker/compose/releases/download/$DOCKER_COMPOSE_VERSION/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose

  # Basic firewall setup
  sudo yum install -y iptables-services
  sudo systemctl start iptables
  sudo systemctl enable iptables
  sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
  sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
  sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
  sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
  sudo iptables -A INPUT -j DROP
  sudo service iptables save

  exit
EOF

# Copy all project files and directories except ledger-key-pair.pem
rsync -av --exclude="$KEY_NAME.pem" -e "ssh -i $KEY_NAME.pem" ./ ec2-user@$INSTANCE_PUBLIC_IP:~/project/

# SSH into the instance and deploy the application
ssh -i "$KEY_NAME.pem" ec2-user@$INSTANCE_PUBLIC_IP << EOF
  cd ~/project
  docker-compose -f docker-compose.prod.yml up -d
  exit
EOF

# Create a hosted zone (if not already created)
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name $DOMAIN_NAME --query 'HostedZones[0].Id' --output text --region $AWS_REGION)

# Create an A record for app.getledger.com pointing to the EC2 instance
cat > change-batch.json << EOF
{
  "Comment": "Creating A record for $SUBDOMAIN.$DOMAIN_NAME",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "$SUBDOMAIN.$DOMAIN_NAME",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [
          {
            "Value": "$INSTANCE_PUBLIC_IP"
          }
        ]
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets --hosted-zone-id $HOSTED_ZONE_ID --change-batch file://change-batch.json

# # Create CNAME record for getledger.com pointing to Netlify
# cat > netlify-change-batch.json << EOF
# {
#   "Comment": "Creating CNAME record for getledger.com pointing to Netlify",
#   "Changes": [
#     {
#       "Action": "UPSERT",
#       "ResourceRecordSet": {
#         "Name": "$DOMAIN_NAME",
#         "Type": "CNAME",
#         "TTL": 300,
#         "ResourceRecords": [
#           {
#             "Value": "$NETLIFY_TARGET"
#           }
#         ]
#       }
#     }
#   ]
# }
# EOF

# aws route53 change-resource-record-sets --hosted-zone-id $HOSTED_ZONE_ID --change-batch file://netlify-change-batch.json

# # Create CNAME record for docs.getledger.com pointing to Mintlify
# cat > mintlify-change-batch.json << EOF
# {
#   "Comment": "Creating CNAME record for docs.getledger.com pointing to Mintlify",
#   "Changes": [
#     {
#       "Action": "UPSERT",
#       "ResourceRecordSet": {
#         "Name": "docs.$DOMAIN_NAME",
#         "Type": "CNAME",
#         "TTL": 300,
#         "ResourceRecords": [
#           {
#             "Value": "$MINTLIFY_TARGET"
#           }
#         ]
#       }
#     }
#   ]
# }
# EOF

# aws route53 change-resource-record-sets --hosted-zone-id $HOSTED_ZONE_ID --change-batch file://mintlify-change-batch.json

# echo "Domain setup complete. Access your app at http://$SUBDOMAIN.$DOMAIN_NAME"

# # Note: For the marketing site on Netlify and documentation on Mintlify,
# # ensure you have set up the appropriate CNAME records in Route 53:
# # - CNAME for getledger.com pointing to your Netlify site
# # - CNAME for docs.getledger.com pointing to your Mintlify site