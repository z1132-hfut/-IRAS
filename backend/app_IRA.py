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
# from IntelligentRecruitmentAssistant.rag.KnowledgeSystem import RAGSystem
# from IntelligentRecruitmentAssistant.knowledge_graph.kg_Q1 import Neo4jQuery

from llm.funetuning_Q1 import LLMInferenceQ1
from llm.funetuning_Q2 import LLMInferenceQ2
from llm.data_clean_Q1 import DataCleanQ1
from llm.data_clean_Q2 import DataCleanQ2
from rag.KnowledgeSystem import RAGSystem
from knowledge_graph.kg_Q1 import Neo4jQuery

import logging
import os
import fitz  # PyMuPDF

# FastAPI应用初始化：
app = FastAPI(
    title="IRAS",
)

# 初始化组件：
# rag_system = RAGSystem()
llm_inference_Q1 = LLMInferenceQ1()
print("Q1模型加载中...")
llm_inference_Q1.load_model()   # 加载模型，最耗时
print("Q1模型加载完成")
llm_inference_Q2 = LLMInferenceQ2()
data_clean_Q1 = DataCleanQ1()
data_clean_Q2 = DataCleanQ2()
rag = RAGSystem()
# Neo4j Aura云托管，无需本地下载：
neo4j_query = Neo4jQuery("neo4j+s://9f26c0e6.databases.neo4j.io", "neo4j",
                         "Fe-28Cvu4lm-WX_03PI5bZcN8jilzWPcDxlgNxfPODo")

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
        job_description: str = Form(..., description="职位描述文本"),
        filter_conditions: str = Form(..., description="JSON格式的字符串")
):
    """
    :param files:上传的PDF文件列表
    :param job_description:岗位需求描述
    :param filter_conditions:自定义条件
    :return:列表，包括传入的全部pdf文件的名称，评分和原因。
    """
    filter_conditions = json.loads(filter_conditions)
    global file
    if filter_conditions is None:
        filter_conditions = {"college_level": 0.6, "internship_experience": 0.9, "project_experience": 0.8, "college_experience": 0.2,
         "self_estimation": 0.2, "others": 0.0, "special_request": "本科及以上，1-3年工作经验"}

    print("filter_conditions: ", filter_conditions)

    prompts = []

    # user_prompt = ("你是一个专业的简历评分助手，你的职责是帮助互联网公司的校招简历筛选。请根据给出的以下信息对简历进行打分。"
    #                "请严格按照以下格式回复：分数：xxx（对简历的打分。0-10分）。原因：xxx（打分原因）。"
    #                "岗位招聘需求：" + str(job_description) +"。信息权重和备注（分别为大学信息，实习经历，项目经历，"
    #                "校内经历，个人介绍，其他内容，备注/特殊需求）："+str(filter_conditions))

    user_prompt = (
        "你是一位专业的校招简历评分专家，需要为互联网公司的AI技术岗位进行简历初筛。"

        "## 评分任务\n"
        "请基于以下信息对简历进行综合评估，重点考察候选人与目标岗位的匹配度。\n"

        "## 评分规则\n"
        "- **总分范围**：0-10分，7分以上为推荐面试，5-7分可备选，5分以下不推荐\n"
        "- **评分依据**：结合岗位需求匹配度 + 权重配置进行加权评估\n"
        "- **特殊要求**：必须检查特殊要求项，如不满足应显著影响分数\n"

        "## 岗位需求\n"
        +str(job_description)+"\n"

        "## 权重配置（分别为大学信息，实习经历，项目经历，校内经历，个人介绍，其他内容，备注/特殊需求。从0-1，权重越接近1越重要）\n"
        +str(filter_conditions)+"\n"

        "## 评估要点\n"
        "1. **技术匹配度**：编程语言、框架、工具是否满足岗位要求\n"
        "2. **项目相关性**：项目经验是否与AI应用开发相关\n"
        "3. **经验完整性**：实习/项目/比赛经历的质量和深度\n"
        "4. **特殊要求**：是否满足备注中的硬性条件\n"

        "## 输出格式（必须严格遵循）\n"
        "分数：x.x（精确到小数点后1位）\n"
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


    for file in files:
        try:
            # 读取PDF文件内容
            pdf_content = await file.read()

            # pdf完整文本
            # full_text = await DataCleanQ1.extract_full_text(data_clean_Q1,pdf_content)

            # 调用数据处理模块
            processed_data = await data_clean_Q1.process_pdf(
                pdf_content,
                file.filename,
            )
            # prompts.append(user_prompt+"。简历内容："+full_text)
            prompts.append(user_prompt+"。简历内容："+str(processed_data))

        except Exception as e:
            prompts.append(str({
                "filename": file.filename,
                "reason": f"处理文件时出错: {str(e)}",
            }))
        finally:
            await file.seek(0)  # 重置文件指针以便后续使用

# 更好地评估基本信息，掌握技术，教育背景，实习经历，项目经历，校内/社会实践经历，获奖/证书信息，自我评价，其他信息

    # rag_system接入知识库——返回：大学等级信息，实习公司信息，实习岗位信息，行业术语信息（技术对比）
    # prompts.append("知识库检索内容：")


    # neo4j_query接入知识图谱
    # prompts.append("知识图谱检索内容：")


    # llm_inference_Q1调用模型（不需重新加载）
    reply = llm_inference_Q1.generate_response(prompts)

    return [
            {
              "pdfName":file.filename,
              "score+analysis":reply,
              # "prompts":prompts,
            },
    ]




