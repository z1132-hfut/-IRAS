from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm
import os

class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """创建唯一性约束"""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT unique_skill IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE")
            session.run("CREATE CONSTRAINT unique_job IF NOT EXISTS FOR (j:Job) REQUIRE j.title IS UNIQUE")
            session.run("CREATE CONSTRAINT unique_candidate IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE")

    def build_from_resumes(self, resume_data_path):
        """从简历数据构建知识图谱"""
        df = pd.read_csv(resume_data_path)

        with self.driver.session() as session:
            # 清空现有数据
            session.run("MATCH (n) DETACH DELETE n")

            # 创建技能节点
            all_skills = set()
            for skills in df['skills'].dropna():
                all_skills.update(skill.strip() for skill in skills.split(','))

            for skill in tqdm(all_skills, desc="Creating Skill nodes"):
                session.run(
                    "MERGE (s:Skill {name: $skill})",
                    skill=skill
                )

            # 创建岗位和候选人节点
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing resumes"):
                # 创建候选人节点
                session.run(
                    """
                    MERGE (c:Candidate {id: $id})
                    SET c.name = $name,
                        c.experience = $experience,
                        c.education = $education
                    """,
                    id=row['id'],
                    name=row['name'],
                    experience=row['experience'],
                    education=row['education']
                )

                # 连接技能关系
                if pd.notna(row['skills']):
                    for skill in row['skills'].split(','):
                        session.run(
                            """
                            MATCH (c:Candidate {id: $id})
                            MATCH (s:Skill {name: $skill})
                            MERGE (c)-[r:HAS_SKILL]->(s)
                            SET r.level = $level
                            """,
                            id=row['id'],
                            skill=skill.strip(),
                            level=row.get('skill_level', 'intermediate')
                        )

                # 连接岗位关系
                if pd.notna(row['target_job']):
                    session.run(
                        """
                        MATCH (c:Candidate {id: $id})
                        MERGE (j:Job {title: $job_title})
                        MERGE (c)-[a:APPLIED_FOR]->(j)
                        SET a.date = $date
                        """,
                        id=row['id'],
                        job_title=row['target_job'],
                        date=row.get('application_date', '2024-01-01')
                    )


if __name__ == "__main__":
    kg_builder = KnowledgeGraphBuilder(
        "neo4j+s://9f26c0e6.databases.neo4j.io",
        "neo4j",
        "Fe-28Cvu4lm-WX_03PI5bZcN8jilzWPcDxlgNxfPODo"
    )
    kg_builder.create_constraints()
    
    # 使用相对路径:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "sample_data.csv")
    data_path = os.path.normpath(data_path)

    kg_builder.build_from_resumes(data_path)
    
    # 新增代码：打印知识图谱
    print("\n当前知识图谱内容：")
    with kg_builder.driver.session() as session:
        # 打印所有节点
        nodes = session.run("MATCH (n) RETURN n")
        print("节点:")
        for node in nodes:
            print(node["n"])
        
        # 打印所有关系
        relationships = session.run("MATCH ()-[r]->() RETURN r")
        print("\n关系:")
        for rel in relationships:
            print(rel["r"])
    
    kg_builder.close()