"""
为岗位能力画像功能提供知识图谱搭建和查询操作。
"""
import os
import pandas as pd
import re
import csv
import time
from neo4j import GraphDatabase
from typing import List, Dict
import jieba
import jieba.posseg as pseg


class KnowledgeGraphBuilder:
    def __init__(self, uri: str, user: str, password: str, data_file: str = None):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=3600,
            connection_timeout=30,
            max_connection_pool_size=50
        )
        # self.data_file = data_file
        self.data_file = "/root/-IRAS/data/data_knowledge_graph/jobInfo.csv"
        self._init_jieba()

    def _init_jieba(self):
        """初始化分词词典"""
        professional_terms = [
            '项目管理', '团队合作', '沟通能力', '问题解决', '领导力', '创新能力',
            '数据分析', '机器学习', '深度学习', '人工智能', '大数据', '云计算',
            '财务分析', '成本控制', '预算管理', '风险管理', '投资分析',
            '市场营销', '品牌策划', '市场调研', '客户关系', '销售技巧',
            '机械设计', '电气自动化', '工艺工程', '质量控制', '生产管理',
            '临床医学', '护理技能', '医疗设备', '药品管理', '病例分析',
            '教育教学', '课程设计', '学生管理', '教育心理学', '教学方法'
        ]

        for term in professional_terms:
            jieba.add_word(term, freq=1000, tag='n')

    def extract_competencies(self, requirement_text: str) -> Dict[str, List[str]]:
        """从职位要求文本中提取各类能力"""
        if pd.isna(requirement_text) or not requirement_text:
            return {
                "technical_skills": [],
                "soft_skills": [],
                "tools_platforms": [],
                "knowledge_domains": []
            }

        competency_dict = {
            "technical_skills": [
                '数据分析', '统计分析', '数据挖掘', '机器学习', '深度学习', '人工智能',
                '编程开发', '软件开发', '系统架构', '数据库设计', '网络管理', '信息安全',
                '机械设计', '电气控制', '自动化技术', '工艺设计', '质量控制', '设备维护',
                '财务分析', '会计核算', '税务筹划', '审计技能', '成本控制', '预算编制',
                '临床技能', '医疗诊断', '护理技术', '手术辅助', '药品配制', '医疗设备操作',
                '教学设计', '课程开发', '教育评估', '心理咨询', '职业指导',
                '市场营销', '品牌管理', '广告策划', '销售技巧', '客户开发', '渠道管理',
                '物流管理', '供应链优化', '仓储管理', '运输规划', '采购谈判',
                '建筑设计', '结构设计', '施工管理', '工程预算', '项目管理',
                '法律咨询', '合同审查', '诉讼代理', '知识产权', '法律研究'
            ],
            "soft_skills": [
                '沟通能力', '表达能力', '团队合作', '协作精神', '领导力', '管理能力',
                '问题解决', '分析能力', '逻辑思维', '创新思维', '创造力', '学习能力',
                '适应能力', '抗压能力', '责任心', '细致认真', '积极主动', '时间管理',
                '决策能力', '规划能力', '组织协调', '谈判技巧', '客户服务', '人际关系'
            ],
            "tools_platforms": [
                'Office', 'Excel', 'Word', 'PowerPoint', 'Outlook', 'WPS',
                'Photoshop', 'Illustrator', 'AutoCAD', 'SolidWorks', '3D Max',
                'Git', 'Docker', 'Kubernetes', 'Jenkins', 'Linux', 'Windows',
                'Python', 'R', 'SQL', 'Tableau', 'Power BI', 'SPSS', 'SAS',
                'SAP', '用友', '金蝶', '广联达', '同花顺', '大智慧'
            ],
            "knowledge_domains": [
                '计算机科学', '信息技术', '电子工程', '机械工程', '土木工程',
                '财务管理', '会计学', '金融学', '经济学', '市场营销', '人力资源管理',
                '临床医学', '护理学', '药学', '中医学', '公共卫生',
                '教育学', '心理学', '法学', '管理学', '物流管理'
            ]
        }

        words = pseg.cut(requirement_text)
        found_competencies = {category: [] for category in competency_dict.keys()}

        for word, flag in words:
            for category, competencies in competency_dict.items():
                for competency in competencies:
                    if competency in word or word in competency:
                        if competency not in found_competencies[category]:
                            found_competencies[category].append(competency)

        for category in found_competencies:
            found_competencies[category] = list(set(found_competencies[category]))

        return found_competencies

    def extract_education_requirement(self, edu_text: str) -> str:
        """提取学历要求"""
        if pd.isna(edu_text) or not edu_text:
            return "不限"

        edu_levels = ['博士', '硕士', '研究生', '本科', '大专', '专科', '中专', '高中', '初中']
        for level in edu_levels:
            if level in edu_text:
                return level
        return "不限"

    def extract_experience_requirement(self, exp_text: str) -> str:
        """提取工作经验要求"""
        if pd.isna(exp_text) or not exp_text:
            return "经验不限"

        exp_patterns = {
            '应届生': r'应届|毕业|无经验|实习生',
            '1年以下': r'1年以下|半年|初级|助理',
            '1-3年': r'1-3年|1年以上|2年|3年|[一二三]年',
            '3-5年': r'3-5年|3年以上|4年|5年|[四五四五]年',
            '5-10年': r'5-10年|5年以上|[六七八九十]年',
            '10年以上': r'10年以上|资深|高级|专家|总监'
        }

        for level, pattern in exp_patterns.items():
            if re.search(pattern, exp_text):
                return level
        return "经验不限"

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 创建副本避免SettingWithCopyWarning
        df_clean = df.copy()

        # 去除重复数据
        df_clean = df_clean.drop_duplicates(subset=['职业', '公司', '要求'])

        # 使用loc避免警告
        df_clean.loc[:, '要求'] = df_clean['要求'].fillna('')
        df_clean.loc[:, '领域'] = df_clean['领域'].fillna('其他')
        df_clean.loc[:, '学历'] = df_clean['学历'].fillna('不限')
        df_clean.loc[:, '经验'] = df_clean['经验'].fillna('经验不限')
        df_clean.loc[:, '城市'] = df_clean['城市'].fillna('未知')

        return df_clean

    def build_knowledge_graph(self, data_file: str = None):
        """从jobInfo.csv构建知识图谱"""
        try:
            # 使用传入的文件路径或默认路径
            if data_file:
                self.data_file = data_file
            else:
                self.data_file="/root/-IRAS/data/data_knowledge_graph/jobInfo.csv"

            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"数据文件不存在: {self.data_file}")

            # 读取数据
            df = pd.read_csv(self.data_file, encoding='utf-8')
            print(f"成功读取数据，共{len(df)}条记录")

            # 数据清洗
            df_clean = self.clean_data(df)
            print("数据清洗完成")

            with self.driver.session() as session:
                # 检查是否已有数据
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]

                if node_count > 0:
                    # print("知识图谱中已有数据，不再重新构建")
                    # return
                    session.run("MATCH (n) DETACH DELETE n")
                    print(f"图谱中已有{node_count}个节点，清空重建...")

                # 创建约束
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Competency) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Education) REQUIRE e.level IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (exp:Experience) REQUIRE exp.level IS UNIQUE"
                ]

                for constraint in constraints:
                    session.run(constraint)

                # 创建学历和经验节点
                education_levels = ['博士', '硕士', '本科', '大专', '中专', '高中', '初中', '不限']
                experience_levels = ['应届生', '1年以下', '1-3年', '3-5年', '5-10年', '10年以上', '经验不限']

                for edu_level in education_levels:
                    session.run("MERGE (e:Education {level: $level})", level=edu_level)

                for exp_level in experience_levels:
                    session.run("MERGE (exp:Experience {level: $level})", level=exp_level)

                # 批量处理数据
                batch_size = 50
                processed_count = 0

                for i in range(0, len(df_clean), batch_size):
                    batch_df = df_clean.iloc[i:i + batch_size]

                    for _, row in batch_df.iterrows():
                        if self._process_job_record(session, row):
                            processed_count += 1

                    print(f"已处理 {min(i + batch_size, len(df_clean))}/{len(df_clean)} 条记录")

                    # 避免资源限制
                    if i % 500 == 0 and i > 0:
                        time.sleep(0.5)

                print(f"知识图谱构建完成，共处理{processed_count}个职位")

        except Exception as e:
            print(f"构建知识图谱时出错: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_job_record(self, session, row) -> bool:
        """处理单个职位记录"""
        if pd.isna(row.get('职业')) or not row.get('职业'):
            return False

        job_name = str(row['职业']).strip()
        if not job_name:
            return False

        education = self.extract_education_requirement(row.get('学历', ''))
        experience = self.extract_experience_requirement(row.get('经验', ''))
        city = str(row.get('城市', '未知')).strip()

        # 创建职位节点
        session.run("""
            MERGE (j:Job {name: $job_name})
            SET j.city = $city, j.industry = $industry
        """, parameters={
            'job_name': job_name,
            'city': city,
            'industry': str(row.get('领域', '')).strip()
        })

        # 关联学历和经验要求
        session.run("""
            MATCH (j:Job {name: $job_name}), (e:Education {level: $education})
            MERGE (j)-[:REQUIRES_EDUCATION]->(e)
        """, parameters={'job_name': job_name, 'education': education})

        session.run("""
            MATCH (j:Job {name: $job_name}), (exp:Experience {level: $experience})
            MERGE (j)-[:REQUIRES_EXPERIENCE]->(exp)
        """, parameters={'job_name': job_name, 'experience': experience})

        # 处理能力要求
        require_content = row.get('要求', '')
        if pd.notna(require_content) and require_content:
            competencies = self.extract_competencies(str(require_content))

            for category, skills in competencies.items():
                for skill in skills:
                    if skill:
                        session.run("""
                            MERGE (c:Competency {name: $skill_name})
                            SET c.category = $category, c.type = 'ability'
                        """, parameters={'skill_name': skill, 'category': category})

                        session.run("""
                            MATCH (j:Job {name: $job_name}), (c:Competency {name: $skill_name})
                            MERGE (j)-[r:REQUIRES_COMPETENCY]->(c)
                            SET r.frequency = coalesce(r.frequency, 0) + 1
                        """, parameters={'job_name': job_name, 'skill_name': skill})

        return True

    def print_knowledge_graph(self):
        """打印展示知识图谱全部内容"""
        try:
            with self.driver.session() as session:
                # 查询所有节点和关系
                query = """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN 
                    labels(n) as node_labels, 
                    properties(n) as node_props,
                    type(r) as rel_type,
                    labels(m) as related_node_labels,
                    properties(m) as related_node_props
                ORDER BY labels(n)[0], n.name
                """

                result = session.run(query)
                records = list(result)

                print("=== 知识图谱内容展示 ===")
                print(f"总节点数: {len(set([str(r['node_props']) for r in records]))}")

                # 按节点类型分组显示
                nodes_by_type = {}
                relationships = []

                for record in records:
                    node_type = record['node_labels'][0] if record['node_labels'] else 'Unknown'
                    node_props = record['node_props']

                    if node_type not in nodes_by_type:
                        nodes_by_type[node_type] = []

                    if node_props and node_props not in [n['props'] for n in nodes_by_type[node_type]]:
                        nodes_by_type[node_type].append({
                            'props': node_props,
                            'relationships': []
                        })

                    if record['rel_type']:
                        relationships.append({
                            'from': node_props,
                            'to': record['related_node_props'],
                            'type': record['rel_type']
                        })

                # 打印节点信息
                for node_type, nodes in nodes_by_type.items():
                    print(f"\n--- {node_type} 节点 ({len(nodes)}个) ---")
                    for i, node in enumerate(nodes[:10]):  # 限制显示数量避免过长
                        print(f"  {i + 1}. {node['props']}")
                    if len(nodes) > 10:
                        print(f"  ... 还有{len(nodes) - 10}个节点")

                # 打印关系信息
                print(f"\n--- 关系 ({len(relationships)}个) ---")
                for i, rel in enumerate(relationships[:10]):
                    from_name = rel['from'].get('name', rel['from'].get('level', 'Unknown'))
                    to_name = rel['to'].get('name', rel['to'].get('level', 'Unknown'))
                    print(f"  {i + 1}. {from_name} --[{rel['type']}]--> {to_name}")
                if len(relationships) > 10:
                    print(f"  ... 还有{len(relationships) - 10}个关系")

                # 打印统计信息
                stats_query = """
                MATCH (j:Job) RETURN count(j) as job_count
                UNION ALL
                MATCH (c:Competency) RETURN count(c) as competency_count
                UNION ALL
                MATCH (e:Education) RETURN count(e) as education_count
                UNION ALL
                MATCH (exp:Experience) RETURN count(exp) as experience_count
                """
                stats_result = session.run(stats_query)
                stats = list(stats_result)

                print(f"\n=== 知识图谱统计 ===")
                print(f"职位节点: {stats[0]['job_count']}")
                print(f"能力节点: {stats[1]['competency_count']}")
                print(f"学历节点: {stats[2]['education_count']}")
                print(f"经验节点: {stats[3]['experience_count']}")

        except Exception as e:
            print(f"打印知识图谱时出错: {e}")
            import traceback
            traceback.print_exc()


class JobCompetencyQuery:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_job_competency_profile(self, job_title: str) -> Dict:
        """获取岗位能力画像"""
        try:
            with self.driver.session() as session:
                # 查询基本要求
                basic_query = """
                MATCH (j:Job {name: $job_title})-[:REQUIRES_EDUCATION]->(e:Education)
                MATCH (j)-[:REQUIRES_EXPERIENCE]->(exp:Experience)
                RETURN e.level AS education, exp.level AS experience, 
                       j.city AS city, j.industry AS industry
                """
                basic_result = session.run(basic_query, parameters={'job_title': job_title})
                basic_record = basic_result.single()

                # 查询能力要求
                competency_categories = ["technical_skills", "soft_skills", "tools_platforms", "knowledge_domains"]
                competency_profile = {}

                for category in competency_categories:
                    category_query = """
                    MATCH (j:Job {name: $job_title})-[:REQUIRES_COMPETENCY]->(c:Competency {category: $category})
                    RETURN c.name AS competency, count(*) AS frequency
                    ORDER BY frequency DESC
                    """
                    category_result = session.run(category_query, parameters={
                        'job_title': job_title,
                        'category': category
                    })
                    competency_profile[category] = [
                        {"skill": record["competency"], "frequency": record["frequency"]}
                        for record in category_result
                    ]

                # 计算摘要信息
                summary_data = self._calculate_competency_summary(session, job_title)

                return {
                    "job_title": job_title,
                    "basic_requirements": {
                        "education": basic_record["education"] if basic_record else "未知",
                        "experience": basic_record["experience"] if basic_record else "未知",
                        "common_cities": [basic_record["city"]] if basic_record and basic_record["city"] else [],
                        "industry": basic_record["industry"] if basic_record else "未知"
                    },
                    "competency_profile": competency_profile,
                    "competency_summary": summary_data
                }

        except Exception as e:
            print(f"查询岗位能力画像时出错: {e}")
            return self._create_empty_profile(job_title)

    def _calculate_competency_summary(self, session, job_title: str) -> Dict:
        """计算能力摘要"""
        summary_query = """
        MATCH (j:Job {name: $job_title})-[:REQUIRES_COMPETENCY]->(c:Competency)
        RETURN count(DISTINCT c) as total_skills, 
               collect(DISTINCT c.category) as categories
        """
        summary_result = session.run(summary_query, parameters={'job_title': job_title})
        summary_record = summary_result.single()

        total_skills = summary_record["total_skills"] if summary_record else 0
        categories = summary_record["categories"] if summary_record else []

        # 最需求能力
        most_demanded_query = """
        MATCH (j:Job {name: $job_title})-[r:REQUIRES_COMPETENCY]->(c:Competency)
        RETURN c.name AS skill, r.frequency AS frequency
        ORDER BY r.frequency DESC LIMIT 1
        """
        most_demanded_result = session.run(most_demanded_query, parameters={'job_title': job_title})
        most_demanded_record = most_demanded_result.single()

        skill_diversity = min(len(set(categories)) / 4 * 100, 100) if categories else 0

        return {
            "total_skills": total_skills,
            "most_demanded": most_demanded_record["skill"] if most_demanded_record else "无",
            "skill_diversity": round(skill_diversity, 2)
        }

    def _create_empty_profile(self, job_title: str) -> Dict:
        """创建空的能力画像"""
        return {
            "job_title": job_title,
            "basic_requirements": {},
            "competency_profile": {
                "technical_skills": [],
                "soft_skills": [],
                "tools_platforms": [],
                "knowledge_domains": []
            },
            "competency_summary": {
                "total_skills": 0,
                "most_demanded": "无",
                "skill_diversity": 0
            }
        }

    def get_job_requirements(self, job_title: str) -> Dict:
        """传入岗位名称，返回需要的能力描述（和学历，工作经验对应）"""
        profile = self.get_job_competency_profile(job_title)

        # 格式化输出
        result = {
            "岗位名称": job_title,
            "学历要求": profile["basic_requirements"].get("education", "未知"),
            "工作经验要求": profile["basic_requirements"].get("experience", "未知"),
            "能力要求": {}
        }

        # 整理能力要求
        for category, skills in profile["competency_profile"].items():
            if skills:
                category_name = {
                    "technical_skills": "技术技能",
                    "soft_skills": "软技能",
                    "tools_platforms": "工具平台",
                    "knowledge_domains": "知识领域"
                }.get(category, category)

                result["能力要求"][category_name] = [skill["skill"] for skill in skills]

        return result


def main():
    """主函数"""
    # Neo4j Aura连接配置
    NEO4J_URI = "neo4j+s://ed7b8137.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "JL8jiUY_gHvv9T2jVveGVPWpU6Od3IG7FumADh4vp2k"

    # 测试连接
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Neo4j Aura连接成功")
        driver.close()
    except Exception as e:
        print(f"Neo4j Aura连接失败: {e}")
        return

    # 修改1：正确设置数据文件路径
    # 获取当前文件所在目录的父目录（项目根目录）
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # 项目根目录
    # DATA_FILE = os.path.join(PROJECT_ROOT, "data_knowledge_graph", "jobInfo.csv")
    DATA_FILE = "/root/-IRAS/data/data_knowledge_graph/jobInfo.csv"
    print(f"数据文件路径: {DATA_FILE}")

    # 检查文件是否存在
    if not os.path.exists(DATA_FILE):
        print(f"错误：数据文件不存在于 {DATA_FILE}")
        # 尝试其他可能的路径
        alternative_paths = [
            os.path.join(PROJECT_ROOT, "data", "data_knowledge_graph", "jobInfo.csv"),
            os.path.join(PROJECT_ROOT, "jobInfo.csv"),
            "jobInfo.csv"
        ]

        for path in alternative_paths:
            if os.path.exists(path):
                DATA_FILE = path
                print(f"使用备选路径: {DATA_FILE}")
                break
        else:
            print("未找到jobInfo.csv文件，请检查文件位置")
            return

    try:
        # 构建知识图谱
        builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_FILE)
        builder.build_knowledge_graph()

        # 打印知识图谱内容
        builder.print_knowledge_graph()

        # 查询测试
        query_tool = JobCompetencyQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # 测试查询几个岗位
        test_jobs = ["数据分析师", "运维工程师", "销售岗"]

        for job in test_jobs:
            print(f"\n=== {job} 能力要求查询结果 ===")
            requirements = query_tool.get_job_requirements(job)

            print(f"岗位名称: {requirements['岗位名称']}")
            print(f"学历要求: {requirements['学历要求']}")
            print(f"工作经验要求: {requirements['工作经验要求']}")
            print("能力要求:")
            for category, skills in requirements['能力要求'].items():
                print(f"  {category}: {', '.join(skills)}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()