"""
负责从RAG中检索相关信息并返回，辅助模型判断
"""

import os
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class KnowledgeSystem:
    def __init__(self):
        # 路径设置
        current_file_path = os.path.abspath(__file__)
        rag_dir = os.path.dirname(current_file_path)
        project_root = os.path.dirname(rag_dir)
        self.data_dir = os.path.join(project_root, "data", "data_knowledge_graph")
        self.vector_store_path = os.path.join(project_root, "rag", "vector_store")

        self.vector_store = None
        self.embeddings = None
        self.knowledge_data = {}

        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """初始化文本嵌入模型"""
        print("正在加载嵌入模型...")
        self.embeddings = HuggingFaceEmbeddings(
            # model_name=r"H:\models\bge-small-zh",
            model_name="/root/models/bge-small-zh",
            # model_kwargs={'device': 'cpu'},
            model_kwargs={'device': 'cuda:0'},
        )
        print("嵌入模型加载完成！")

    def load_data_files(self) -> Dict[str, List[Document]]:
        """加载所有数据文件"""
        documents_dict = {}

        # 1. 加载大学排名数据
        college_csv_path = os.path.join(self.data_dir, "Chinese-colleges.csv")
        if os.path.exists(college_csv_path):
            college_docs = self._parse_college_csv(college_csv_path)
            documents_dict["colleges"] = college_docs
            self.knowledge_data["colleges_df"] = pd.read_csv(college_csv_path, encoding='utf-8')
            print(f"加载大学数据: {len(college_docs)} 条记录")

        # 2. 加载大学经历数据
        experience_path = os.path.join(self.data_dir, "college_experience.txt")
        if os.path.exists(experience_path):
            experience_docs, experience_dict = self._parse_experience_data(experience_path)
            documents_dict["experiences"] = experience_docs
            self.knowledge_data["experiences"] = experience_dict
            print(f"加载大学经历数据: {len(experience_docs)} 个文档")

        # 3. 加载招聘偏好数据
        preference_path = os.path.join(self.data_dir, "recruit_preference.txt")
        if os.path.exists(preference_path):
            preference_docs, preference_dict = self._parse_preference_data(preference_path)
            documents_dict["preferences"] = preference_docs
            self.knowledge_data["preferences"] = preference_dict
            print(f"加载招聘偏好数据: {len(preference_docs)} 个文档")

        return documents_dict

    def _parse_college_csv(self, file_path: str) -> List[Document]:
        """解析大学排名CSV文件"""
        documents = []
        df = pd.read_csv(file_path, encoding='utf-8')

        for _, row in df.iterrows():
            college_name = row['学校名称'].strip()
            content_parts = [
                f"大学名称: {college_name}",
                f"排名: {row['排名']}",
                f"省市: {row['省市']}",
                f"学校类型: {row['学校类型']}",
                f"是否双一流: {row['是否双一流']}",
                f"是否985: {row['是否985']}",
                f"是否211: {row['是否211']}",
                f"总分: {row['总分']}"
            ]

            content = "\n".join(content_parts)
            document = Document(
                page_content=content,
                metadata={
                    "type": "college",
                    "college_name": college_name,
                    "ranking": row['排名']
                }
            )
            documents.append(document)

        return documents

    def _parse_experience_data(self, file_path: str) -> Tuple[List[Document], Dict]:
        """解析大学经历数据"""
        documents = []
        experience_dict = {}

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按章节分割
        sections = re.split(r'一、|二、|三、|四、', content)
        experience_types = ["学术科研经历", "学科竞赛经历", "社会实践与领导力经历", "企业相关实践经历"]

        for i, section in enumerate(sections[1:], 1):
            section_title = experience_types[i - 1]
            experience_dict[section_title] = {}

            # 按行处理
            lines = section.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 跳过表头
                if any(keyword in line for keyword in ['经历名称', '竞赛类别', '描述', '含金量']):
                    continue

                # 处理制表符分隔的数据
                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 2:
                        experience_name = parts[0]
                        description = parts[1]
                        criteria = parts[2] if len(parts) > 2 else ""

                        experience_dict[section_title][experience_name] = {
                            "description": description,
                            "criteria": criteria
                        }

                        content = f"{section_title}\n经历名称: {experience_name}\n描述: {description}"
                        if criteria:
                            content += f"\n含金量标准: {criteria}"

                        document = Document(
                            page_content=content,
                            metadata={
                                "type": "experience",
                                "category": section_title,
                                "experience_name": experience_name
                            }
                        )
                        documents.append(document)

        return documents, experience_dict

    def _parse_preference_data(self, file_path: str) -> Tuple[List[Document], Dict]:
        """解析招聘偏好数据 - 完全重写以正确解析完整特征"""
        documents = []
        preference_dict = {
            "企业类型": {},
            "岗位类型": {},
            "行业类型": {}
        }

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按大章节分割
        sections = re.split(r'一、\s*企业类型维度|二、\s*岗位类型维度|三、\s*不同行业招聘能力偏好特征', content)

        # 1. 解析企业类型维度
        if len(sections) > 1:
            company_section = sections[1]
            lines = company_section.split('\n')
            current_category = ""
            current_features = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('分类') or '招聘偏好特征' in line:
                    continue

                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 2:
                        left_part = parts[0]
                        right_part = parts[1]

                        # 检查是否是企业类型名称（不是以数字开头）
                        if not re.match(r'^\d\.', left_part) and left_part:
                            # 保存上一个类别的特征
                            if current_category and current_features:
                                preference_dict["企业类型"][current_category] = "\n".join(current_features)

                            # 开始新的类别
                            current_category = left_part
                            current_features = [right_part]
                        else:
                            # 累积特征
                            if current_category:
                                current_features.append(right_part)

            # 保存最后一个类别的特征
            if current_category and current_features:
                preference_dict["企业类型"][current_category] = "\n".join(current_features)

        # 2. 解析岗位类型维度
        if len(sections) > 2:
            position_section = sections[2]
            lines = position_section.split('\n')
            current_category = ""
            current_features = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('分类') or '招聘偏好特征' in line:
                    continue

                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 2:
                        left_part = parts[0]
                        right_part = parts[1]

                        if not re.match(r'^\d\.', left_part) and left_part:
                            if current_category and current_features:
                                preference_dict["岗位类型"][current_category] = "\n".join(current_features)

                            current_category = left_part
                            current_features = [right_part]
                        else:
                            if current_category:
                                current_features.append(right_part)

            if current_category and current_features:
                preference_dict["岗位类型"][current_category] = "\n".join(current_features)

        # 3. 解析行业类型维度
        if len(sections) > 3:
            industry_section = sections[3]
            lines = industry_section.split('\n')
            current_category = ""
            current_features = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('行业分类') or '核心能力偏好' in line or '具体说明' in line:
                    continue

                if '\t' in line:
                    parts = [part.strip() for part in line.split('\t') if part.strip()]
                    if len(parts) >= 2:
                        left_part = parts[0]
                        right_part = parts[1]

                        # 行业名称通常不以数字开头，且不是"偏好原因"等说明性文字
                        if (not re.match(r'^\d\.', left_part) and left_part and
                                not any(keyword in left_part for keyword in ['偏好原因', '看重证据', '-'])):
                            if current_category and current_features:
                                preference_dict["行业类型"][current_category] = "\n".join(current_features)

                            current_category = left_part
                            current_features = [right_part]
                        else:
                            if current_category:
                                current_features.append(right_part)

            if current_category and current_features:
                preference_dict["行业类型"][current_category] = "\n".join(current_features)

        # 创建文档
        for category_type, categories in preference_dict.items():
            for name, features in categories.items():
                if features.strip():
                    # 清理特征文本，确保格式正确
                    cleaned_features = []
                    for feature in features.split('\n'):
                        feature = feature.strip()
                        if feature:
                            # 确保特征以数字编号开头
                            if not re.match(r'^\d\.', feature):
                                # 尝试修复格式
                                if cleaned_features and re.match(r'^\d\.', cleaned_features[-1]):
                                    # 如果上一行是编号行，当前行可能是续行
                                    cleaned_features[-1] += " " + feature
                                else:
                                    cleaned_features.append(feature)
                            else:
                                cleaned_features.append(feature)

                    final_features = "\n".join(cleaned_features)
                    content = f"{category_type}: {name}\n招聘偏好特征：\n{final_features}"

                    document = Document(
                        page_content=content,
                        metadata={
                            "type": "preference",
                            "category": category_type,
                            "subcategory": name
                        }
                    )
                    documents.append(document)

        return documents, preference_dict

    def preprocess_documents(self, documents_dict: Dict[str, List[Document]]) -> List[Document]:
        """预处理所有文档"""
        all_documents = []

        for doc_type, docs in documents_dict.items():
            all_documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )

        print("正在进行文本分割...")
        split_docs = text_splitter.split_documents(all_documents)
        print(f"分割完成：{len(all_documents)} -> {len(split_docs)} 个文本块")

        return split_docs

    def build_knowledge_base(self, force_rebuild: bool = False):
        """构建知识库"""
        if not force_rebuild and os.path.exists(self.vector_store_path):
            print("加载已存在的向量知识库...")
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("知识库加载完成！")
            return

        print("开始构建新的知识库...")
        documents_dict = self.load_data_files()

        if not documents_dict:
            raise ValueError("没有加载到任何数据！")

        split_documents = self.preprocess_documents(documents_dict)

        print("正在创建向量索引...")
        self.vector_store = FAISS.from_documents(split_documents, self.embeddings)

        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"知识库构建完成！已保存到 {self.vector_store_path}")

    def batch_search(self, search_terms: List[str], top_k: int = 3) -> List[str]:
        """批量搜索相关信息"""
        if self.vector_store is None:
            raise ValueError("知识库未初始化，请先调用 build_knowledge_base()")

        results = []

        for term in search_terms:
            if not term.strip():
                continue

            # 首先尝试精确匹配
            exact_match = self._exact_match_search(term)
            if exact_match:
                results.append(exact_match)
                continue

            # 如果没有精确匹配，使用相似性搜索
            try:
                query = self._build_query(term)
                search_results = self.vector_store.similarity_search(query, k=top_k)
                formatted_result = self._format_similarity_result(term, search_results)
                results.append(formatted_result)

            except Exception as e:
                print(f"搜索 '{term}' 时出错: {e}")
                results.append(f"未找到关于'{term}'的相关信息。")

        return results

    def _exact_match_search(self, term: str) -> Optional[str]:
        """精确匹配搜索"""
        term_lower = term.lower().strip()

        # 1. 检查大学名称
        if "colleges_df" in self.knowledge_data:
            df = self.knowledge_data["colleges_df"]

            # 精确匹配
            exact_match = df[df['学校名称'].str.strip() == term]
            if not exact_match.empty:
                row = exact_match.iloc[0]
                return self._format_college_result(row)

            # 包含匹配
            contains_match = df[df['学校名称'].str.contains(term, na=False)]
            if not contains_match.empty:
                row = contains_match.iloc[0]
                return self._format_college_result(row)

        # 2. 检查招聘偏好
        if "preferences" in self.knowledge_data:
            prefs = self.knowledge_data["preferences"]

            # 检查企业类型
            for category, features in prefs.get("企业类型", {}).items():
                if term_lower in category.lower():
                    return f"{category}招聘偏好特征：\n{features.strip()}"

            # 检查岗位类型
            for position, features in prefs.get("岗位类型", {}).items():
                if term_lower in position.lower():
                    return f"{position}招聘偏好特征：\n{features.strip()}"

            # 检查行业类型
            for industry, features in prefs.get("行业类型", {}).items():
                if term_lower in industry.lower():
                    return f"{industry}能力偏好特征：\n{features.strip()}"

        # 3. 检查经历类型
        if "experiences" in self.knowledge_data:
            experiences = self.knowledge_data["experiences"]

            for category, exp_dict in experiences.items():
                for exp_name, exp_info in exp_dict.items():
                    if term_lower in exp_name.lower():
                        desc = exp_info.get("description", "")
                        criteria = exp_info.get("criteria", "")
                        result = f"{exp_name}经历信息：\n描述: {desc}"
                        if criteria:
                            result += f"\n含金量标准: {criteria}"
                        return result

        return None

    def _format_college_result(self, row) -> str:
        """格式化大学搜索结果"""
        return f"{row['学校名称']}基本信息：\n排名: {row['排名']}\n省市: {row['省市']}\n学校类型: {row['学校类型']}\n是否双一流: {row['是否双一流']}\n是否985: {row['是否985']}\n是否211: {row['是否211']}\n总分: {row['总分']}"

    def _build_query(self, term: str) -> str:
        """根据搜索词类型构建查询"""
        term_lower = term.lower()

        if any(keyword in term_lower for keyword in ['大学', '学院', '学校']):
            return f"{term} 大学 高校 教育"
        elif any(keyword in term_lower for keyword in ['国企', '私企', '外企', '企业']):
            return f"{term} 企业类型 招聘 偏好"
        elif any(keyword in term_lower for keyword in
                 ['技术', '研发', '销售', '市场', '设计', '产品', '运营', '人力资源', '财务']):
            return f"{term} 岗位 职位 招聘要求"
        else:
            return term

    def _format_similarity_result(self, term: str, results: List[Document]) -> str:
        """格式化相似性搜索结果"""
        if not results:
            return f"未找到关于'{term}'的相关信息。"

        response_parts = [f"【{term}】的相关信息："]
        seen_content = set()

        for i, doc in enumerate(results, 1):
            content = doc.page_content.strip()

            if content in seen_content:
                continue
            seen_content.add(content)

            response_parts.append(f"{i}. {content}")

        if len(response_parts) == 1:
            return f"未找到关于'{term}'的精确匹配信息。"

        return "\n".join(response_parts[:4])


def main():
    """测试函数"""
    kb = KnowledgeSystem()
    kb.build_knowledge_base(force_rebuild=True)  # 强制重建以确保数据正确解析

    print("\n" + "=" * 50)
    print("智能招聘知识库系统")
    print("=" * 50)

    # 测试精确匹配
    test_terms = ["互联网/科技行业","高端制造/硬科技行业","金融行业",
                  "北京大学", "合肥工业大学",
                  "国有企业", "外资企业", "初创企业",
                  "技术研发类", "产品运营类", "人力资源类",
                  ]
    print(f"\n测试搜索: {test_terms}")

    results = kb.batch_search(test_terms)

    for i, result in enumerate(results):
        print(f"\n{'-' * 30}")
        print(f"结果 {i + 1}:")
        print(result)

    # 交互式搜索
    while True:
        print("\n请输入搜索词（多个词用逗号分隔，输入'quit'退出）：")
        user_input = input().strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("感谢使用！")
            break

        if not user_input:
            continue

        search_terms = [term.strip() for term in user_input.split(',')]
        results = kb.batch_search(search_terms)

        print("\n" + "=" * 50)
        for i, result in enumerate(results):
            print(f"\n搜索结果 {i + 1}:")
            print(result)
            print("-" * 30)


if __name__ == "__main__":
    main()