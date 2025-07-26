from fastapi import FastAPI, HTTPException   # FastAPI相关。FastAPI：用于创建API的现代、快速（高性能）的Web框架
from pydantic import BaseModel   # 数据模型验证。BaseModel: Pydantic的基础模型类，用于定义数据模型和进行数据验证
from typing import List  # 类型提示。List: 用于类型注解，表示列表类型（Python的类型提示系统）
# 项目自定义模块：
from IntelligentRecruitmentAssistant.llm.inference import LLMInference
from IntelligentRecruitmentAssistant.rag.retrieval import RAGSystem
from IntelligentRecruitmentAssistant.knowledge_graph.neo4j_utils import Neo4jQuery
# 系统工具库：
import logging   # logging: Python标准日志模块，用于记录应用程序的运行日志
import os        # 提供与操作系统交互的功能（如环境变量、文件系统操作等）
import ssl       # 用于处理SSL/TLS加密，可能用于安全连接（如数据库连接）

# 设置使用系统证书。配置系统使用默认SSL证书路径，确保HTTPS请求安全
os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(ssl.get_default_verify_paths().openssl_cafile)

# FastAPI运行：
# 命令行：uvicorn IntelligentRecruitmentAssistant.backend.app:app --reload
# 进入网页：http://127.0.0.1:8000/docs

# FastAPI应用初始化：
app = FastAPI(
    title="Intelligent Recruitment Assistant",
    description="AI-powered recruitment system with LLaMA3, RAG and Knowledge Graph"
)

# 初始化组件：
print("初始化组件--初始化RAG文档检索系统")
rag_system = RAGSystem()
print("初始化组件--加载LLM语言模型")
llm_inference = LLMInference()
print("初始化组件--连接Neo4j知识图谱数据库")
neo4j_query = Neo4jQuery("neo4j+s://9f26c0e6.databases.neo4j.io", "neo4j",
                         "Fe-28Cvu4lm-WX_03PI5bZcN8jilzWPcDxlgNxfPODo")

# 请求模型定义：

# 定义简历匹配API的请求体结构：
class ResumeMatchRequest(BaseModel):
    job_description: str
    resume_text: str

# 定义面试问题生成API的请求体结构
class InterviewQuestionRequest(BaseModel):
    job_title: str
    candidate_skills: List[str]

# API端点实现：

# 简历匹配端点：
@app.post("/match-resume")
async def match_resume(request: ResumeMatchRequest):
    """评估简历与岗位的匹配度"""
    try:
        # 1. 从知识库检索相关上下文
        context = rag_system.hybrid_retrieve(request.job_description)

        # 2. 从知识图谱获取相关技能
        skills = neo4j_query.get_job_requirements(request.job_description)

        # 3. 使用LLM评估匹配度
        prompt = f"""
        岗位描述: {request.job_description}，
        所需技能: {', '.join(skills)}，
        简历内容: {request.resume_text}，
        相关上下文: {' '.join(context)}

        请评估这份简历与岗位的匹配度(0-100分)，并给出具体改进建议。
        返回格式: {{"score": 分数, "analysis": "分析文本", "suggestions": ["建议1", "建议2"]}}
        """
        print("提示词：",prompt)
        response = llm_inference.generate(prompt)
        print("回复：",response)
        return response
    except Exception as e:
        logging.error(f"Error in match_resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate-interview-questions")
# async def generate_interview_questions(request: InterviewQuestionRequest):
#     """生成基于岗位和技能的面试问题"""
#     try:
#         # 1. 从知识图谱获取相关知识点
#         knowledge = neo4j_query.get_skill_path(
#             request.job_title,
#             str(request.candidate_skills)
#         )
#
#         # 2. 使用LLM生成问题
#         prompt = f"""
#         岗位: {request.job_title}
#         候选人技能: {', '.join(request.candidate_skills)}
#         相关知识: {knowledge}
#
#         生成5个技术面试问题和3个行为面试问题，按STAR法则给出理想答案要点。
#         返回JSON格式: {{"technical_questions": [...], "behavioral_questions": [...]}}
#         """
#         print("提示词：",prompt)
#         response = llm_inference.generate(prompt)
#         print("回复：",response)
#         return response
#     except Exception as e:
#         logging.error(f"Error in generate_interview_questions: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-interview-questions")
async def generate_interview_questions(request: InterviewQuestionRequest):
    """生成基于岗位和技能的面试问题"""
    try:
        # 1. 从知识图谱获取相关技能信息
        skills_info = neo4j_query.get_related_skills(
            request.job_title,
            request.candidate_skills
        )

        # 准备知识文本
        knowledge_text = "岗位通常需要的技能:\n"
        knowledge_text += "\n".join([
            f"- {skill['name']}: {skill.get('description', '无描述')}"
            for skill in skills_info.get("skills_info", [])
        ])

        knowledge_text += "\n\n候选人已具备的技能: " + ", ".join(skills_info.get("candidate_skills", []))
        knowledge_text += "\n\n候选人需要提升的技能: " + ", ".join(skills_info.get("missing_skills", []))

        # 2. 使用LLM生成问题
        prompt = f"""
        岗位名称: {request.job_title}
        候选人技能: {', '.join(request.candidate_skills)}

        相关知识信息:
        {knowledge_text}

        请根据以上信息:
        1. 生成5个针对候选人已具备技能的技术深度问题
        2. 生成3个针对候选人需要提升技能的学习潜力问题
        3. 生成2个行为面试问题(考察团队合作和问题解决能力)

        返回JSON格式: {{
            "technical_questions": [
                {{
                    "question": "问题内容",
                    "skill": "相关技能",
                    "purpose": "考察目的"
                }},
                ...
            ],
            "potential_questions": [
                {{
                    "question": "问题内容", 
                    "skill": "相关需要提升的技能",
                    "purpose": "考察学习能力"
                }},
                ...
            ],
            "behavioral_questions": [
                {{
                    "question": "问题内容",
                    "purpose": "考察点"
                }},
                ...
            ],
            "analysis": "整体匹配度分析"
        }}
        """
        print("提示词：", prompt)
        response = llm_inference.generate(prompt)
        print("回复：", response)
        return response
    except Exception as e:
        logging.error(f"Error in generate_interview_questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import asyncio
    
    # 测试简历匹配功能
    async def test_match_resume():
        test_request = ResumeMatchRequest(
            job_description="需要3年以上Python开发经验，熟悉FastAPI和机器学习",
            resume_text="我有5年Python开发经验，使用过FastAPI和TensorFlow"
        )
        result = await match_resume(test_request)
        print("简历匹配测试结果:", result)
    
    # 测试面试问题生成功能
    async def test_generate_questions():
        test_request = InterviewQuestionRequest(
            job_title="Python开发工程师",
            candidate_skills=["Python", "FastAPI", "机器学习"]
        )
        result = await generate_interview_questions(test_request)
        print("面试问题生成测试结果:", result)
    
    # 运行测试
    async def main():
        await test_match_resume()
        await test_generate_questions()
    
    asyncio.run(main())
    

