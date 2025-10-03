"""
后端入口程序
云服务器地址：ssh root@8.149.138.59
FastAPI运行：
命令行：uvicorn backend.app_IRA:app --reload
进入网页：http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import json
import re

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List, Dict, Any
import fitz  # PyMuPDF

# from IntelligentRecruitmentAssistant.llm.funetuning_Q1 import LLMInferenceQ1
# from IntelligentRecruitmentAssistant.llm.funetuning_Q2 import LLMInferenceQ2
# from IntelligentRecruitmentAssistant.llm.data_clean_Q1 import DataCleanQ1
# from IntelligentRecruitmentAssistant.llm.data_clean_Q2 import DataCleanQ2
# from IntelligentRecruitmentAssistant.rag.KnowledgeSystem import KnowledgeSystem
# from IntelligentRecruitmentAssistant.knowledge_graph.kg_Q1 import KnowledgeGraphBuilder,JobCompetencyQuery

from llm.funetuning_Q1 import LLMInferenceQ1
from llm.funetuning_Q2 import LLMInferenceQ2
from llm.data_clean_Q1 import DataCleanQ1
from llm.data_clean_Q2 import DataCleanQ2
from rag.KnowledgeSystem import KnowledgeSystem
from knowledge_graph.kg_Q1 import KnowledgeGraphBuilder,JobCompetencyQuery

import logging
import os
import fitz  # PyMuPDF

# FastAPI应用初始化：
app = FastAPI(
    title="IRAS",
)

# 初始化组件：
# rag_system = RAGSystem()
llm_inference_Q1_uf = LLMInferenceQ1()
print("Q1模型加载中...")
llm_inference_Q1_uf.load_model()   # 加载模型，耗时
print("Q1模型微调中...")
llm_inference_Q1_uf.finetune_with_qlora(
        data_path="/root/-IRAS/data/data_model_train/QLora_data.txt",
        output_dir="/root/models/finetuned_model",
        num_train_epochs=3
    )
print("Q1模型微调完成。正在加载微调后的模型")
llm_inference_Q1 = LLMInferenceQ1("/root/models/finetuned_model")
llm_inference_Q1.load_model()
print("微调后模型加载完成")

llm_inference_Q2 = LLMInferenceQ2()
data_clean_Q1 = DataCleanQ1()
# data_clean_Q2 = DataCleanQ2()
rag = KnowledgeSystem()
rag.build_knowledge_base()  # 不强制重建

# Neo4j Aura云托管，无需本地下载：
neo4j_query_kg = KnowledgeGraphBuilder("neo4j+s://ed7b8137.databases.neo4j.io", "neo4j",
                         "JL8jiUY_gHvv9T2jVveGVPWpU6Od3IG7FumADh4vp2k").build_knowledge_graph()

neo4j_query = JobCompetencyQuery("neo4j+s://ed7b8137.databases.neo4j.io", "neo4j",
                         "JL8jiUY_gHvv9T2jVveGVPWpU6Od3IG7FumADh4vp2k")


class FilterConditions(BaseModel):
    college_level: Optional[float] = 0.6
    internship_experience: Optional[float] = 0.9
    project_experience: Optional[float] = 0.8
    college_experience: Optional[float] = 0.2
    self_estimation: Optional[float] = 0.2
    others: Optional[float] = 0.2
    special_request: Optional[str] = "本科及以上，1-3年工作经验"


@app.post("/match-resume")
async def match_resumes(
        files: List[UploadFile] = File(..., description="批量上传的PDF简历文件"),
        company_kind: str = Form(..., description="公司类型"),
        job_kind: str = Form(..., description="岗位类型"),
        trade_kind: str = Form(..., description="行业类型"),
        job_name: str = Form(..., description="职位名称"),
        job_description: str = Form(..., description="职位描述文本"),
        company_description: str = Form(..., description="公司描述文本"),
        filter_conditions: str = Form(..., description="JSON格式的字符串")
):
    """
    简历匹配：支持对pdf格式的简历进行批量解析并筛选，解析完成后给出pdf简历名称+评分+解释。
    :param trade_kind: 行业类型
    :param job_kind: 岗位类型
    :param company_kind: 公司类型
    :param company_description:公司描述
    :param job_name:职位名称
    :param files:上传的PDF文件列表
    :param job_description:岗位需求描述
    :param filter_conditions:自定义条件
    :return:列表，包括传入的全部pdf文件名称，评分和原因。
    """
    filter_conditions = json.loads(filter_conditions)
    global file
    if filter_conditions is None:
        filter_conditions = {"college_level": 0.4, "internship_experience": 0.5, "project_experience": 0.8,
                "college_experience": 0.2,"self_estimation": 0.2, "others": 0.1, "special_request": "暂无"}

    print("filter_conditions: ", filter_conditions)

    prompts = []

    # 批量获取数据库查询信息
    kg_info = rag.batch_search([company_kind,job_kind,trade_kind])

    # 每次提问前加在建立内容之前的话
    user_prompt = (
        "你是一位专业的简历评分专家，需要为公司的招聘岗位进行简历初筛。"
        "## 评分任务\n"
        "请基于以下信息对简历进行综合评估，重点考察候选人与目标岗位的匹配度。\n"
        "## 评分规则\n"
        "- **总分范围**：0-100分，75分以上为推荐面试，55-75分可备选，55分以下不推荐\n"
        "- **评分依据**：结合岗位需求匹配度 + 权重配置进行加权评估\n"
        "- **特殊要求**：必须检查特殊要求项，如不满足应显著影响分数\n"
        "## 公司类型及其招聘偏好：\n"
        +str(kg_info[0])+"\n"
        "## 岗位类型及其招聘偏好：\n"
        +str(kg_info[1])+"\n"
        "## 行业类型及其招聘偏好：\n"
        +str(kg_info[2])+"\n"
        "## 岗位名称\n"
        +str(job_name)+"\n"
        "## 公司相关信息\n"
        +str(company_description)+"\n"                  
        "## 岗位需求\n"
        +str(job_description)+"\n"
        "## 岗位能力画像：\n"
        +str(neo4j_query.get_job_requirements(job_name))+"\n"                      
        "## 权重配置（分别为大学信息，实习经历，项目经历，校内经历，个人介绍，其他内容，备注/特殊需求。从0-1，权重越接近1越重要）\n"
        +str(filter_conditions)+"\n"
        "## 评估要点\n"
        "1. **技能匹配度**：面试者拥有的技能是否满足岗位要求\n"
        "2. **经历相关性**：面试者的经历（实习，项目等）是否与岗位需求相关\n"
        "3. **其他内容质量**：面试者其他经历（校内外社会实践，竞赛获奖，证书，论文等等）的质量和深度\n"
        "4. **特殊要求**：是否满足备注中的硬性条件\n"
        "## 输出格式（必须严格遵循）\n"
        "分数：x.x（精确到小数点后2位）\n"
        "原因：\n"
        "- 匹配优势：列出符合岗位要求的亮点（至少3项）\n"
        "- 不足之处：列出缺失或不符合的关键点（至少2项）\n"
        "- 权重影响：说明各项权重对最终分数的影响\n"
        "- 总体评价：给出是否推荐面试的建议\n"
        "## 注意事项\n"
        "- 评分要客观公正，避免过度宽松或严格\n"
        "- 对于校招简历，适当放宽经验要求，重点考察技术潜力和学习能力\n"
        "- 特殊要求不满足时，应在原因中明确说明并影响分数"
    )

    file_names = []

    for file in files:
        try:
            # 读取PDF文件内容
            pdf_content = await file.read()

            # pdf完整文本
            # full_text = await DataCleanQ1.extract_full_text(data_clean_Q1,pdf_content)

            file_names.append(file.filename)

            # 调用数据处理模块
            processed_data = await data_clean_Q1.process_pdf(
                pdf_content,
                file.filename,
            )

            prompts.append(user_prompt+"。##简历内容："+str(processed_data))

        except Exception as e:
            prompts.append(str({
                "filename": file.filename,
                "reason": f"处理文件时出错: {str(e)}",
            }))
        finally:
            await file.seek(0)  # 重置文件指针以便后续使用

    # llm_inference_Q1调用模型（不需重新加载）
    replies = llm_inference_Q1.generate_response(prompts)

    # 列表推导式。
    return [{"file_name":file_name, "score_analysis":reply} for file_name,reply in zip(file_names, replies)]



@app.post("/question_generate")
async def match_resumes(
        files: List[UploadFile] = File(..., description="批量上传的PDF简历文件"),
        company_kind: str = Form(..., description="公司类型"),
        job_kind: str = Form(..., description="岗位类型"),
        trade_kind: str = Form(..., description="行业类型"),
        job_name: str = Form(..., description="职位名称"),
        job_description: str = Form(..., description="职位描述文本"),
        company_description: str = Form(..., description="公司描述文本"),
        questions_num: int = Form(..., description="面试问题数量"),
        questions_request: str = Form(..., description="面试问题要求")
):
    """
    面试问题生成：（数据集倾向于IT/互联网行业）
    针对pdf格式的简历生成一系列面试问题，可指定问题数量和设置问题生成提示词。
    :param trade_kind: 行业类型
    :param job_kind: 岗位类型
    :param company_kind: 公司类型
    :param company_description:公司描述
    :param job_name:职位名称
    :param files:上传的PDF文件列表
    :param job_description:岗位需求描述
    :param questions_num:面试问题数量
    :param questions_request:面试问题要求
    :return:列表，包括传入的全部pdf文件的名称和生成的面试问题。
    """

