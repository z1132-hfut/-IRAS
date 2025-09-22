"""
针对简历匹配功能对本地的Qwen2.5-3B-Instruct进行微调。
优化版本：模型只加载一次，支持多次提问
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os
from typing import List, Optional

# 设置环境变量，优化CPU运行
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 确保不使用GPU


class LLMInferenceQ1:
    def __init__(self, model_path: str = r"H:\models\Qwen2.5-3B-Instruct"):
        """
        初始化模型
        :param model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """
        加载模型和tokenizer
        """
        if self.is_loaded:
            return

        print("正在加载模型和tokenizer...")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 加载模型（使用8位量化）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                load_in_8bit=True,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print("模型加载完成！")

        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def generate_response(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        生成回复（核心方法，可多次调用）
        :param prompts: 提示词列表
        :param generation_kwargs: 生成参数
        :return: 回复列表
        """
        if not self.is_loaded:
            self.load_model()

        replies = []

        # 设置默认生成参数
        default_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        default_kwargs.update(generation_kwargs)

        for prompt in prompts:
            try:
                # 构建消息格式
                messages = [{"role": "user", "content": prompt}]

                # 编码输入
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )

                # 生成输出
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs,
                        **default_kwargs
                    )

                # 解码结果（跳过特殊token）
                response = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                # 移除输入部分，只保留生成的回复
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

                replies.append(response)
                replies.append("\n")

            except Exception as e:
                print(f"生成过程中出错: {e}")
                replies.append(f"错误: {str(e)}")

        return replies



# 使用示例
if __name__ == "__main__":
    # 初始化模型
    llm = LLMInferenceQ1(r"H:\models\Qwen2.5-3B-Instruct")

    # 第一次使用时会自动加载模型
    prompts = [
        "请分析这份简历的技术栈匹配度",
        "评估这份简历的工作经验相关性"
    ]

    # 第一次调用（包含模型加载时间）
    results1 = llm.generate_response(prompts)
    print("第一次结果:", results1)

    # 第二次使用：
    prompts2 = [
        "你是谁",
        "今天合肥的天气怎么样"
    ]

    # 第二次调用（不包含模型加载时间）
    results2 = llm.generate_response(prompts2)
    print("第二次结果:", results2)

