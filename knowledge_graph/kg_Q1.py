"""
为简历匹配功能提供知识图谱搭建和查询操作。
"""

"""
基于知识图谱进行复杂的关联查询和智能分析
"""
from neo4j import GraphDatabase
from typing import Optional, List, Dict

class Neo4jQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_related_skills(self, job_title: str, candidate_skills: List[str]) -> Dict:
        """
        获取与岗位和候选人技能相关的知识
        返回格式: {
            "job_skills": [岗位相关技能],
            "candidate_skills": [候选人已有技能],
            "missing_skills": [岗位需要但候选人缺少的技能]
        }
        """
        try:
            with self.driver.session() as session:
                # 查询岗位通常需要的技能(通过分析已申请该岗位的候选人技能)
                job_skills_query = """
                MATCH (j:Job {title: $title})<-[:APPLIED_FOR]-(c:Candidate)-[:HAS_SKILL]->(s:Skill)
                RETURN collect(DISTINCT s.name) as skills
                """
                job_result = session.run(job_skills_query, title=job_title)
                job_skills = job_result.single()["skills"] if job_result.single() else []

                # 筛选候选人已有技能
                matched_skills = [skill for skill in candidate_skills if skill in job_skills]
                missing_skills = [skill for skill in job_skills if skill not in candidate_skills]

                # 获取技能描述(如果有)
                skills_info_query = """
                UNWIND $skills AS skill_name
                MATCH (s:Skill {name: skill_name})
                RETURN collect({name: s.name, description: s.description}) as skills_info
                """
                skills_info = session.run(skills_info_query, skills=job_skills).single()
                skills_info = skills_info["skills_info"] if skills_info else []

                return {
                    "job_skills": job_skills,
                    "candidate_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "skills_info": skills_info
                }
        except Exception as e:
            print(f"Error querying related skills: {e}")
            return {
                "job_skills": [],
                "candidate_skills": [],
                "missing_skills": [],
                "skills_info": []
            }

    def get_job_requirements(self, job_title: str) -> List[str]:
        """获取岗位的技能要求(通过分析已申请该岗位的候选人技能)"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (j:Job {title: $title})<-[:APPLIED_FOR]-(c:Candidate)-[:HAS_SKILL]->(s:Skill)
                    RETURN collect(DISTINCT s.name) as skills
                    """, title=job_title)
                record = result.single()
                return record["skills"] if record else []
        except Exception as e:
            print(f"Error querying job requirements: {e}")
            return []

