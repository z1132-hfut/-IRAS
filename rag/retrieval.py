import os
from pathlib import Path
from typing import List
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re


class RAGSystem:
    def __init__(self, data_path=None):
        """
        初始化RAG系统

        参数:
            data_path: 文档目录路径
        """
        # 1. 设置文档路径
        self.data_path = self._resolve_data_path(data_path)
        os.makedirs(self.data_path, exist_ok=True)

        # 2. 初始化核心组件
        self._initialize_components()

        # 3. 加载文档并创建向量存储
        try:
            documents = self.load_documents()
            if not documents:
                raise ValueError("未加载到有效文档")

            # 关键修复：创建向量存储
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        except Exception as e:
            logging.error(f"初始化失败: {str(e)}")
            raise

    def _resolve_data_path(self, data_path: str) -> str:
        """解析文档路径"""
        if data_path is None:
            base_dir = Path(__file__).parent.parent.parent
            return str(base_dir / "IntelligentRecruitmentAssistant" / "data" / "job_descriptions")
        return data_path

    def _initialize_components(self):
        """初始化各组件"""
        # 1. 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # 2. 嵌入模型
        try:
            self.embeddings = OllamaEmbeddings(
                model="deepseek-r1:1.5b",
                base_url="http://localhost:11434",
            )
            # 测试嵌入
            test_embed = self.embeddings.embed_query("测试")
            if not test_embed:
                raise ValueError("嵌入测试失败")
        except Exception as e:
            raise RuntimeError(f"嵌入模型初始化失败: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        text = re.sub(r'[\uE000-\uF8FF]', '', text)  # 移除特殊符号
        return re.sub(r'javascript:\S+|收藏立即沟通|举报', '', text)  # 移除网页元素

    def load_documents(self) -> List[Document]:
        """加载并清洗文档"""
        documents = []

        for filename in os.listdir(self.data_path):
            filepath = os.path.join(self.data_path, filename)

            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = self._clean_text(f.read())
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": filename}
                            ))

                elif filename.endswith(('.docx', '.doc')):
                    from unstructured.partition.auto import partition
                    elements = partition(filepath)
                    text = "\n".join([str(el) for el in elements])
                    documents.append(Document(
                        page_content=self._clean_text(text),
                        metadata={"source": filename}
                    ))

            except Exception as e:
                logging.warning(f"跳过文件 {filename}: {str(e)}")

        return documents

    def hybrid_retrieve(self, query: str, k: int = 3) -> List[str]:
        """混合检索（返回字符串内容列表）

        参数:
            query: 查询文本
            k: 返回结果数量

        返回:
            文档内容字符串列表
        """
        if not hasattr(self, 'vector_store'):
            raise RuntimeError("请先调用load_documents()初始化向量存储")

        try:
            # 语义检索
            semantic = self.vector_store.similarity_search(query, k=k)
            # 关键词检索
            keyword = self.vector_store.max_marginal_relevance_search(query, k=k)
            # 合并去重
            combined = {doc.page_content: doc for doc in semantic}
            combined.update({doc.page_content: doc for doc in keyword})

            # 返回纯文本内容列表
            return [doc.page_content for doc in combined.values()][:k]

        except Exception as e:
            raise RuntimeError(f"检索失败: {str(e)}")



if __name__ == "__main__":
    try:
        # 测试初始化
        rag = RAGSystem()
        print("RAG系统初始化成功！")

        # 测试检索
        results = rag.hybrid_retrieve("Java开发")
        print(f"检索到 {len(results)} 条结果:")
        for result in results:
            print(result)

    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        print("排查建议:")
        print("1. 检查Ollama服务是否运行 (ollama serve)")
        print("2. 确认模型已下载 (ollama pull deepseek-r1:1.5b)")
        print("3. 查看data/job_descriptions目录是否有文档")