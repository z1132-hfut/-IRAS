"""
负责从RAG中检索相关信息并返回，辅助模型判断
"""

import os
import re
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAGSystem:
    # def __init__(self, data_file: str = "collegeInfo.txt", vector_store_path: str = "vector_store"):
    #     self.data_file = data_file
    #     self.vector_store_path = vector_store_path
    #     self.vector_store = None
    #     self.embeddings = None
    #
    #     # 初始化嵌入模型
    #     self._initialize_embeddings()
    #
    # def _initialize_embeddings(self):
    #     """初始化文本嵌入模型"""
    #     print("正在加载嵌入模型...")
    #     self.embeddings = HuggingFaceEmbeddings(
    #         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #         model_kwargs={'device': 'cpu'}
    #     )
    #     print("嵌入模型加载完成！")
    #
    # def parse_college_data(self) -> List[Document]:
    #     """解析大学数据文件，返回Document列表"""
    #     documents = []
    #
    #     if not os.path.exists(self.data_file):
    #         raise FileNotFoundError(f"数据文件 {self.data_file} 不存在！")
    #
    #     print(f"正在解析数据文件: {self.data_file}")
    #
    #     with open(self.data_file, 'r', encoding='utf-8') as file:
    #         content = file.read()
    #
    #     # 按大学分割内容（假设每个大学信息以大学名称开头）
    #     # 这里根据实际文件格式调整分割逻辑
    #     college_blocks = self._split_college_blocks(content)
    #
    #     for i, block in enumerate(college_blocks):
    #         if block.strip():
    #             # 提取大学名称作为metadata
    #             college_name = self._extract_college_name(block)
    #
    #             document = Document(
    #                 page_content=block.strip(),
    #                 metadata={
    #                     "college_name": college_name,
    #                     "source": self.data_file,
    #                     "block_id": i
    #                 }
    #             )
    #             documents.append(document)
    #
    #     print(f"成功解析 {len(documents)} 个大学信息块")
    #     return documents
    #
    # def _split_college_blocks(self, content: str) -> List[str]:
    #     """分割大学信息块 - 根据实际文件格式调整"""
    #     # 方法1: 按空行分割（如果每个大学信息之间有空行）
    #     blocks = re.split(r'\n\s*\n', content)
    #
    #     # 方法2: 如果文件有特定格式，比如以"大学名称："开头
    #     # blocks = re.split(r'(?=^大学名称：|^校名：|^【)', content, flags=re.MULTILINE)
    #
    #     # 过滤空块
    #     return [block for block in blocks if block.strip()]
    #
    # def _extract_college_name(self, block: str) -> str:
    #     """从信息块中提取大学名称"""
    #     # 尝试多种模式匹配大学名称
    #     patterns = [
    #         r'^大学名称：\s*([^\n]+)',
    #         r'^校名：\s*([^\n]+)',
    #         r'^【([^】]+)】',
    #         r'^([^\s：]+大学)[\s：]',
    #         r'^([^\s：]+学院)[\s：]'
    #     ]
    #
    #     for pattern in patterns:
    #         match = re.search(pattern, block, re.MULTILINE)
    #         if match:
    #             return match.group(1).strip()
    #
    #     # 如果没匹配到，返回第一行作为名称
    #     first_line = block.split('\n')[0].strip()
    #     return first_line if first_line else "未知大学"
    #
    # def preprocess_documents(self, documents: List[Document]) -> List[Document]:
    #     """预处理文档：文本分割"""
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=500,  # 每个块500字符
    #         chunk_overlap=50,  # 重叠50字符
    #         length_function=len,
    #         separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    #     )
    #
    #     print("正在进行文本分割...")
    #     split_docs = text_splitter.split_documents(documents)
    #     print(f"分割完成：{len(documents)} -> {len(split_docs)} 个文本块")
    #
    #     return split_docs
    #
    # def build_knowledge_base(self, force_rebuild: bool = False):
    #     """构建知识库"""
    #     # 检查是否已存在向量存储
    #     if not force_rebuild and os.path.exists(self.vector_store_path):
    #         print("加载已存在的向量知识库...")
    #         self.vector_store = FAISS.load_local(
    #             self.vector_store_path,
    #             self.embeddings,
    #             allow_dangerous_deserialization=True
    #         )
    #         print("知识库加载完成！")
    #         return
    #
    #     # 构建新的知识库
    #     print("开始构建新的知识库...")
    #
    #     # 1. 解析数据
    #     documents = self.parse_college_data()
    #
    #     if not documents:
    #         raise ValueError("没有解析到任何大学数据！")
    #
    #     # 2. 预处理和分割
    #     split_documents = self.preprocess_documents(documents)
    #
    #     # 3. 创建向量存储
    #     print("正在创建向量索引...")
    #     self.vector_store = FAISS.from_documents(split_documents, self.embeddings)
    #
    #     # 4. 保存向量存储
    #     os.makedirs(self.vector_store_path, exist_ok=True)
    #     self.vector_store.save_local(self.vector_store_path)
    #     print(f"知识库构建完成！已保存到 {self.vector_store_path}")
    #
    # def search_college_info(self, college_name: str, top_k: int = 3) -> str:
    #     """搜索大学信息"""
    #     if self.vector_store is None:
    #         raise ValueError("知识库未初始化，请先调用 build_knowledge_base()")
    #
    #     # 构建搜索查询
    #     query = f"{college_name} 大学 信息 简介"
    #
    #     print(f"搜索大学: {college_name}")
    #
    #     # 执行相似性搜索
    #     try:
    #         # 方法1: 直接相似性搜索
    #         results = self.vector_store.similarity_search(query, k=top_k)
    #
    #         # 方法2: 带metadata过滤的搜索（更精确）
    #         # results = self.vector_store.similarity_search(
    #         #     query,
    #         #     k=top_k,
    #         #     filter={"college_name": college_name}  # 根据实际metadata结构调整
    #         # )
    #
    #     except Exception as e:
    #         print(f"搜索出错: {e}")
    #         # 尝试更宽松的搜索
    #         results = self.vector_store.similarity_search(college_name, k=top_k)
    #
    #     if not results:
    #         return f"未找到关于'{college_name}'的相关信息。"
    #
    #     # 整理和汇总结果
    #     return self._format_search_results(college_name, results)
    #
    # def _format_search_results(self, college_name: str, results: List[Document]) -> str:
    #     """格式化搜索结果"""
    #     response_parts = [f"关于【{college_name}】的搜索结果：\n"]
    #
    #     seen_content = set()  # 避免重复内容
    #
    #     for i, doc in enumerate(results, 1):
    #         content = doc.page_content.strip()
    #
    #         # 去重
    #         if content in seen_content:
    #             continue
    #         seen_content.add(content)
    #
    #         # 清理和格式化内容
    #         cleaned_content = re.sub(r'\s+', ' ', content)  # 去除多余空白
    #         cleaned_content = cleaned_content.replace(college_name, f"**{college_name}**")
    #
    #         response_parts.append(f"{i}. {cleaned_content}\n")
    #
    #     # 如果没找到直接匹配的结果，返回最相关的结果
    #     if len(response_parts) == 1:
    #         response_parts.append("未找到完全匹配的信息，以下是最相关的信息：\n")
    #         for i, doc in enumerate(results[:2], 1):
    #             content = doc.page_content.strip()[:200] + "..."  # 截断长文本
    #             response_parts.append(f"{i}. {content}\n")
    #
    #     return "\n".join(response_parts)
    #
    # def get_all_college_names(self) -> List[str]:
    #     """获取知识库中所有大学的名称"""
    #     if self.vector_store is None:
    #         return []
    #
    #     # 从metadata中提取所有大学名称
    #     college_names = set()
    #     # 注意：这里需要根据实际的存储方式调整
    #     # 如果是FAISS，可能需要遍历所有文档的metadata
    #
    #     return list(college_names)


# def create_sample_data():
#     """创建示例数据文件（如果不存在）"""
#     sample_data = """
# 大学名称：清华大学
# 所在地：北京市
# 创办时间：1911年
# 简介：清华大学是中国著名高等学府，坐落于北京西北郊风景秀丽的清华园，是国家级重点大学。
#
# 大学名称：北京大学
# 所在地：北京市
# 创办时间：1898年
# 简介：北京大学创立于1898年维新变法之际，初名京师大学堂，是中国近现代第一所国立综合性大学。
#
# 大学名称：浙江大学
# 所在地：浙江省杭州市
# 创办时间：1897年
# 简介：浙江大学是一所特色鲜明、在海内外有较大影响的综合型、研究型、创新型大学。
#
# 大学名称：复旦大学
# 所在地：上海市
# 创办时间：1905年
# 简介：复旦大学是中国人自主创办的第一所高等院校，是一所世界知名、国内顶尖的综合性研究型大学。
#
# 大学名称：上海交通大学
# 所在地：上海市
# 创办时间：1896年
# 简介：上海交通大学是我国历史最悠久、享誉海内外的高等学府之一，是教育部直属并与上海市共建的全国重点大学。
# """
#
#     if not os.path.exists("collegeInfo.txt"):
#         with open("collegeInfo.txt", "w", encoding="utf-8") as f:
#             f.write(sample_data)
#         print("已创建示例数据文件: collegeInfo.txt")
#
#
# def main():
#     """主函数"""
#     # 创建示例数据（如果不存在）
#     create_sample_data()
#
#     # 初始化知识库系统
#     kb = RAGSystem()
#
#     # 构建知识库（如果已存在则直接加载）
#     kb.build_knowledge_base(force_rebuild=False)  # 设置为True可强制重新构建
#
#     print("\n" + "=" * 50)
#     print("大学信息检索系统")
#     print("=" * 50)
#
#     # 交互式搜索
#     while True:
#         print("\n请输入大学名称进行搜索（输入'quit'退出）：")
#         user_input = input().strip()
#
#         if user_input.lower() in ['quit', 'exit', '退出']:
#             print("感谢使用！")
#             break
#
#         if not user_input:
#             continue
#
#         # 执行搜索
#         result = kb.search_college_info(user_input)
#         print("\n" + "=" * 50)
#         print(result)
#         print("=" * 50)
#
#
# if __name__ == "__main__":
#     main()
    pass




