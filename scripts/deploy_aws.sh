#!/bin/bash
# 部署到AWS EC2
EC2_INSTANCE="your-instance-id"
EC2_KEY="your-key.pem"

# 复制文件
scp -i $EC2_KEY -r . ec2-user@$EC2_INSTANCE:/home/ec2-user/recruitment-assistant

# 启动服务
ssh -i $EC2_KEY ec2-user@$EC2_INSTANCE << 'EOF'
cd /home/ec2-user/recruitment-assistant
sudo docker-compose up -d --build
EOF