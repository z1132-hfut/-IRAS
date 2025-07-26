# 智能招聘助手系统

## 项目概述
基于LLaMA3-8B微调+知识图谱+RAG的智能招聘解决方案，实现简历自动匹配和面试问题生成。

## 技术栈
- 大模型: LLaMA3-8B + QLoRA微调
- 知识图谱: Neo4j
- 检索增强: LangChain + FAISS
- 后端: FastAPI
- 部署: Docker + AWS EC2

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

## 部署指南
### Docker部署
```bash
docker build -t recruitment-assistant .
docker run -p 8000:8000 recruitment-assistant
```