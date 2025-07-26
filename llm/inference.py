from langchain_community.llms import Ollama
import warnings


class LLMInference:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        """
        初始化 Ollama 模型
        :param model_name: Ollama 模型名称，默认为 deepseek-r1:1.5b
        """
        # 忽略 LangChain 的某些警告（如果需要）
        warnings.filterwarnings("ignore", category=UserWarning)

        # 初始化 Ollama 模型
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",  # Ollama 默认服务地址
            temperature=0.7,  # 控制生成随机性
            top_p=0.9,  # 核采样参数
            num_ctx=2048  # 上下文窗口大小
        )

    def generate(self, prompt: str) -> str:
        """
        生成文本响应
        :param prompt: 输入的提示文本
        :return: 模型生成的响应文本
        """
        # 直接调用 Ollama 模型生成响应
        response = self.llm(prompt)
        return response