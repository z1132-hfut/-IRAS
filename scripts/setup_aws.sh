#!/bin/bash
# 设置AWS环境
AWS_REGION="us-east-1"
EC2_INSTANCE="your-instance-id"
EC2_KEY="your-key.pem"

# 安装必要工具
ssh -i $EC2_KEY ec2-user@$EC2_INSTANCE << 'EOF'
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -a -G docker ec2-user
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
EOF

# 设置Neo4j密码
NEO4J_PASSWORD=$(openssl rand -base64 12)
sed -i "s/your_password/$NEO4J_PASSWORD/g" knowledge_graph/*.py