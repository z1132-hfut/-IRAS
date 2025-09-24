"""
针对简历匹配功能对本地的Qwen2.5-3B-Instruct进行微调。
优化版本：模型只加载一次，支持多次提问
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os
from typing import List, Optional


class LLMInferenceQ1:
    def __init__(self, model_path: str = "/root/models/Qwen2.5-3B-Instruct"):
        """
        初始化模型
        :param model_path: 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        # 自动检测可用设备
        self.device = self._setup_device()
        print(f"使用设备: {self.device}")

    def _setup_device(self):
        """
        设置运行设备，优先使用GPU
        """
        if torch.cuda.is_available():
            # 使用GPU
            gpu_count = torch.cuda.device_count()
            print(f"检测到 {gpu_count} 个GPU:")
            for i in range(gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return torch.device("cuda")
        else:
            # 使用CPU
            print("未检测到GPU，使用CPU运行")
            return torch.device("cpu")

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

            # 根据设备选择不同的加载策略
            if self.device.type == "cuda":
                # GPU加载策略
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",  # 自动分配到可用GPU
                    torch_dtype=torch.float16,  # 使用半精度减少内存占用
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("模型已加载到GPU")
            else:
                # CPU加载策略
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("模型已加载到CPU")

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print("模型加载完成！")

        except Exception as e:
            print(f"模型加载失败: {e}")
            # 如果GPU加载失败，尝试CPU回退
            if "CUDA" in str(e) or "GPU" in str(e):
                print("GPU加载失败，尝试使用CPU...")
                self.device = torch.device("cpu")
                self._load_model_cpu()
            else:
                raise

    def _load_model_cpu(self):
        """CPU回退加载"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            self.is_loaded = True
            print("模型已回退到CPU加载完成！")
        except Exception as e:
            print(f"CPU加载也失败: {e}")
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

        # 设置默认生成参数（根据设备优化）
        default_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # GPU特定优化参数
        if self.device.type == "cuda":
            default_kwargs.update({
                "temperature": 0.8,  # GPU上可以稍微提高创造性
                "top_k": 50,  # 添加top_k过滤
            })

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

                # 将输入移动到对应设备
                if self.device.type == "cuda":
                    inputs = inputs.to(self.device)

                # 生成输出
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs,
                        **default_kwargs
                    )

                # 将结果移回CPU进行解码（避免GPU内存占用）
                if generated_ids.is_cuda:
                    generated_ids = generated_ids.cpu()

                # 解码结果（跳过特殊token）
                response = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                # 移除输入部分，只保留生成的回复
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

                replies.append(response)

            except Exception as e:
                print(f"生成过程中出错: {e}")
                replies.append(f"错误: {str(e)}")

        return replies

    def get_device_info(self):
        """获取设备信息"""
        info = {
            "device_type": self.device.type,
            "model_loaded": self.is_loaded
        }

        if self.device.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB"
            })

        return info

    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("GPU缓存已清理")


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    llm = LLMInferenceQ1("/root/models/Qwen2.5-3B-Instruct")

    # 显示设备信息
    print("设备信息:", llm.get_device_info())

    # 第一次使用时会自动加载模型
    prompts = [
        "请分析这份简历的技术栈匹配度",
        "评估这份简历的工作经验相关性"
    ]

    # 第一次调用（包含模型加载时间）
    results1 = llm.generate_response(prompts)
    print("第一次结果:", results1)

    # 显示GPU内存使用情况
    if llm.device.type == "cuda":
        print("GPU内存使用:", llm.get_device_info()["gpu_memory"])

    # 第二次使用：
    prompts2 = [
        "你是谁",
        "今天合肥的天气怎么样"
    ]

    # 第二次调用（不包含模型加载时间）
    results2 = llm.generate_response(prompts2)
    print("第二次结果:", results2)

    # 清理缓存
    llm.clear_cache()