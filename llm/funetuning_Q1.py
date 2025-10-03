# """
# 针对简历匹配功能对本地的Qwen2.5-3B-Instruct进行微调。
# 优化版本：模型只加载一次，支持多次提问
# 加入QLoRA微调功能
# """
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import torch
# import os
# import json
# from typing import List, Optional
# from datasets import Dataset
# from trl import SFTTrainer
#
#
# class LLMInferenceQ1:
#     def __init__(self, model_path: str = "/root/models/Qwen2.5-3B-Instruct"):
#         """
#         初始化模型
#         :param model_path: 模型路径
#         """
#         self.model_path = model_path
#         self.model = None
#         self.tokenizer = None
#         self.is_loaded = False
#
#         # 自动检测可用设备
#         self.device = self._setup_device()
#         print(f"使用设备: {self.device}")
#
#     def _setup_device(self):
#         """
#         设置运行设备，优先使用GPU
#         """
#         if torch.cuda.is_available():
#             # 使用GPU
#             gpu_count = torch.cuda.device_count()
#             print(f"检测到 {gpu_count} 个GPU:")
#             for i in range(gpu_count):
#                 print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#             return torch.device("cuda")
#         else:
#             # 使用CPU
#             print("未检测到GPU，使用CPU运行")
#             return torch.device("cpu")
#
#     def load_model(self):
#         """
#         加载模型和tokenizer
#         """
#         if self.is_loaded:
#             return
#
#         print("正在加载模型和tokenizer...")
#
#         try:
#             # 加载tokenizer
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_path,
#                 trust_remote_code=True
#             )
#
#             # 根据设备选择不同的加载策略
#             if self.device.type == "cuda":
#                 # GPU加载策略
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     self.model_path,
#                     device_map="auto",  # 自动分配到可用GPU
#                     torch_dtype=torch.float16,  # 使用半精度减少内存占用
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True
#                 )
#                 print("模型已加载到GPU")
#             else:
#                 # CPU加载策略
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     self.model_path,
#                     device_map="cpu",
#                     torch_dtype=torch.float32,
#                     trust_remote_code=True,
#                     low_cpu_mem_usage=True
#                 )
#                 print("模型已加载到CPU")
#
#             # 设置pad_token
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#
#             self.is_loaded = True
#             print("模型加载完成！")
#
#         except Exception as e:
#             print(f"模型加载失败: {e}")
#             # 如果GPU加载失败，尝试CPU回退
#             if "CUDA" in str(e) or "GPU" in str(e):
#                 print("GPU加载失败，尝试使用CPU...")
#                 self.device = torch.device("cpu")
#                 self._load_model_cpu()
#             else:
#                 raise
#
#     def _load_model_cpu(self):
#         """CPU回退加载"""
#         try:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 device_map="cpu",
#                 torch_dtype=torch.float32,
#                 trust_remote_code=True
#             )
#             self.is_loaded = True
#             print("模型已回退到CPU加载完成！")
#         except Exception as e:
#             print(f"CPU加载也失败: {e}")
#             raise
#
#     def generate_response(self, prompts: List[str], **generation_kwargs) -> List[str]:
#         """
#         生成回复（核心方法，可多次调用）
#         :param prompts: 提示词列表
#         :param generation_kwargs: 生成参数
#         :return: 回复列表
#         """
#         if not self.is_loaded:
#             self.load_model()
#
#         replies = []
#
#         # 设置默认生成参数（根据设备优化）
#         default_kwargs = {
#             "max_new_tokens": 1024,
#             "do_sample": True,
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "repetition_penalty": 1.1,
#             "pad_token_id": self.tokenizer.eos_token_id
#         }
#
#         # GPU特定优化参数
#         if self.device.type == "cuda":
#             default_kwargs.update({
#                 "temperature": 0.8,  # GPU上可以稍微提高创造性
#                 "top_k": 50,  # 添加top_k过滤
#             })
#
#         default_kwargs.update(generation_kwargs)
#
#         for prompt in prompts:
#             try:
#                 # 构建消息格式
#                 messages = [{"role": "user", "content": prompt}]
#
#                 # 编码输入
#                 inputs = self.tokenizer.apply_chat_template(
#                     messages,
#                     tokenize=True,
#                     add_generation_prompt=True,
#                     return_tensors="pt"
#                 )
#
#                 # 将输入移动到对应设备
#                 if self.device.type == "cuda":
#                     inputs = inputs.to(self.device)
#
#                 # 生成输出
#                 with torch.no_grad():
#                     generated_ids = self.model.generate(
#                         inputs,
#                         **default_kwargs
#                     )
#
#                 # 将结果移回CPU进行解码（避免GPU内存占用）
#                 if generated_ids.is_cuda:
#                     generated_ids = generated_ids.cpu()
#
#                 # 解码结果（跳过特殊token）
#                 response = self.tokenizer.decode(
#                     generated_ids[0],
#                     skip_special_tokens=True
#                 )
#
#                 # 移除输入部分，只保留生成的回复
#                 if response.startswith(prompt):
#                     response = response[len(prompt):].strip()
#
#                 replies.append(response)
#
#             except Exception as e:
#                 print(f"生成过程中出错: {e}")
#                 replies.append(f"错误: {str(e)}")
#
#         return replies
#
#     def get_device_info(self):
#         """获取设备信息"""
#         info = {
#             "device_type": self.device.type,
#             "model_loaded": self.is_loaded
#         }
#
#         if self.device.type == "cuda":
#             info.update({
#                 "gpu_name": torch.cuda.get_device_name(),
#                 "gpu_memory": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB"
#             })
#
#         return info
#
#     def clear_cache(self):
#         """清理GPU缓存"""
#         if self.device.type == "cuda":
#             torch.cuda.empty_cache()
#             print("GPU缓存已清理")
#
#     def load_training_data(self, data_path: str):
#         """
#         加载训练数据
#         :param data_path: 数据文件路径
#         """
#         try:
#             with open(data_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             print(f"成功加载 {len(data)} 条训练数据")
#             return data
#         except Exception as e:
#             print(f"加载训练数据失败: {e}")
#             return None
#
#     def prepare_data_for_training(self, data, test_size=0.1):
#         """
#         准备训练数据
#         :param data: 原始数据
#         :param test_size: 测试集比例
#         """
#         # 构建训练格式
#         formatted_data = []
#         for item in data:
#             # 提取system和user内容
#             content = item.get("system", "") + item.get("user", "")
#             assistant_content = item.get("assistant", "")
#
#             # 构建训练文本
#             text = f"<|im_start|>system\n{content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
#             formatted_data.append({"text": text})
#
#         # 划分训练集和测试集
#         split_idx = int(len(formatted_data) * (1 - test_size))
#         train_data = formatted_data[:split_idx]
#         test_data = formatted_data[split_idx:]
#
#         print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
#         return train_data, test_data
#
#     def setup_lora_config(self):
#         """
#         设置QLoRA配置
#         """
#         lora_config = LoraConfig(
#             r=16,  # LoRA秩
#             lora_alpha=32,  # LoRA alpha
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#         return lora_config
#
#     def finetune_with_qlora(self,
#                             data_path: str = "/root/-IRAS/data/data_model_train/QLora_data.txt",
#                             output_dir: str = "/root/models/finetuned_model",
#                             num_train_epochs: int = 2,  # 减少到2轮
#                             per_device_train_batch_size: int = 1,
#                             gradient_accumulation_steps: int = 1,
#                             learning_rate: float = 1e-4,
#                             max_seq_length: int = 512):  # 进一步减少序列长度
#         """
#         使用QLoRA进行模型微调（优化版本）
#         """
#         print("开始QLoRA微调...")
#
#         # 加载训练数据
#         raw_data = self.load_training_data(data_path)
#         if raw_data is None:
#             print("无法加载训练数据，终止微调")
#             return
#
#         # 准备数据
#         train_data, eval_data = self.prepare_data_for_training(raw_data)
#
#         print(f"训练数据准备完成: {len(train_data)} 条训练, {len(eval_data)} 条验证")
#
#         # 确保模型已加载
#         if not self.is_loaded:
#             self.load_model()
#
#         # 清理GPU缓存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         # 检查GPU内存，如果不足直接使用CPU
#         if torch.cuda.is_available():
#             free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
#             free_memory_gb = free_memory / (1024 ** 3)
#             print(f"GPU可用内存: {free_memory_gb:.2f} GB")
#
#             if free_memory_gb < 2.0:  # 小于2GB直接使用CPU
#                 print("GPU内存不足，直接使用CPU训练")
#                 self._finetune_on_cpu_optimized(train_data, eval_data, output_dir, num_train_epochs, learning_rate)
#                 return
#
#         try:
#             # GPU训练
#             self._finetune_on_gpu(train_data, eval_data, output_dir, num_train_epochs, learning_rate, max_seq_length)
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower():
#                 print("GPU内存不足，切换到CPU训练...")
#                 self._finetune_on_cpu_optimized(train_data, eval_data, output_dir, num_train_epochs, learning_rate)
#             else:
#                 raise e
#
#     def _finetune_on_gpu(self, train_data, eval_data, output_dir, num_train_epochs, learning_rate, max_seq_length):
#         """在GPU上进行微调"""
#         print("在GPU上进行微调...")
#
#         # 准备模型用于训练
#         self.model = prepare_model_for_kbit_training(self.model)
#
#         # 设置LoRA配置
#         lora_config = LoraConfig(
#             r=8,
#             lora_alpha=16,
#             target_modules=["q_proj", "v_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#
#         # 应用LoRA到模型
#         self.model = get_peft_model(self.model, lora_config)
#         self.model.print_trainable_parameters()
#
#         # 设置训练参数
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=num_train_epochs,
#             per_device_train_batch_size=1,
#             gradient_accumulation_steps=1,
#             learning_rate=learning_rate,
#             logging_steps=5,
#             save_steps=len(train_data),  # 每个epoch保存一次
#             eval_strategy="no",  # 关闭验证以节省内存
#             save_strategy="epoch",
#             report_to=None,
#             remove_unused_columns=True,
#             warmup_ratio=0.1,
#             lr_scheduler_type="linear",
#             optim="adamw_torch",
#             fp16=True,
#             dataloader_pin_memory=False,
#             gradient_checkpointing=True,
#         )
#
#         def formatting_func(example):
#             return example["text"]
#
#         trainer = SFTTrainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=Dataset.from_list(train_data),
#             formatting_func=formatting_func,
#         )
#
#         # 开始训练
#         print("开始GPU训练...")
#         trainer.train()
#
#         # 保存模型
#         trainer.save_model()
#         self.tokenizer.save_pretrained(output_dir)
#         print(f"GPU微调完成，模型保存到: {output_dir}")
#
#         # 清理
#         self.model = self.model.merge_and_unload()
#         torch.cuda.empty_cache()
#
#     def _finetune_on_cpu_optimized(self, train_data, eval_data, output_dir, num_train_epochs, learning_rate):
#         """在CPU上进行优化的微调"""
#         print("在CPU上进行优化微调...")
#
#         # 将模型移动到CPU
#         self.model = self.model.cpu()
#         self.device = torch.device("cpu")
#
#         # 准备模型用于训练
#         self.model = prepare_model_for_kbit_training(self.model)
#
#         # 设置LoRA配置
#         lora_config = LoraConfig(
#             r=4,  # 进一步减少参数
#             lora_alpha=8,
#             target_modules=["q_proj", "v_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type="CAUSAL_LM",
#         )
#
#         # 应用LoRA到模型
#         self.model = get_peft_model(self.model, lora_config)
#         self.model.print_trainable_parameters()
#
#         # 设置训练参数 - CPU优化
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             num_train_epochs=num_train_epochs,
#             per_device_train_batch_size=1,
#             gradient_accumulation_steps=1,
#             learning_rate=learning_rate,
#             logging_steps=1,  # 更频繁的日志
#             save_steps=len(train_data),  # 每个epoch保存
#             eval_strategy="no",
#             save_strategy="epoch",
#             report_to=None,
#             remove_unused_columns=True,
#             warmup_ratio=0.1,
#             lr_scheduler_type="linear",
#             optim="adamw_torch",
#             use_cpu=True,  # 使用新参数
#             dataloader_num_workers=0,  # 避免多进程问题
#             dataloader_prefetch_factor=None,
#         )
#
#         def formatting_func(example):
#             return example["text"]
#
#         trainer = SFTTrainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=Dataset.from_list(train_data),
#             formatting_func=formatting_func,
#         )
#
#         # 开始训练
#         print("开始CPU训练...")
#         trainer.train()
#
#         # 保存模型
#         trainer.save_model()
#         self.tokenizer.save_pretrained(output_dir)
#         print(f"CPU微调完成，模型保存到: {output_dir}")
#
#         # 清理
#         self.model = self.model.merge_and_unload()
#
# # 使用示例
# if __name__ == "__main__":
#     # 初始化模型
#     llm = LLMInferenceQ1("/root/models/Qwen2.5-3B-Instruct")
#
#     # 显示设备信息
#     print("设备信息:", llm.get_device_info())
#
#     # 微调模型（取消注释以运行微调）
#     llm.finetune_with_qlora(
#         data_path="IntelligentRecruitmentAssistant/data/data_model_train/QLora_data.txt",
#         output_dir="IntelligentRecruitmentAssistant/llm/finetuned_model",
#         num_train_epochs=3
#     )
#
#     # 第一次使用时会自动加载模型
#     prompts = [
#         "请分析这份简历的技术栈匹配度",
#         "评估这份简历的工作经验相关性"
#     ]
#
#     # 第一次调用（包含模型加载时间）
#     results1 = llm.generate_response(prompts)
#     print("第一次结果:", results1)
#
#     # 显示GPU内存使用情况
#     if llm.device.type == "cuda":
#         print("GPU内存使用:", llm.get_device_info()["gpu_memory"])
#
#     # 第二次使用：
#     prompts2 = [
#         "你是谁",
#         "今天合肥的天气怎么样"
#     ]
#
#     # 第二次调用（不包含模型加载时间）
#     results2 = llm.generate_response(prompts2)
#     print("第二次结果:", results2)
#
#     # 清理缓存
#     llm.clear_cache()

"""
针对简历匹配功能对本地的Qwen2.5-3B-Instruct进行微调。
优化版本：模型只加载一次，支持多次提问
加入QLoRA微调功能
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import json
from typing import List, Optional
from datasets import Dataset
from trl import SFTTrainer


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
        加载模型和tokenizer（修复版本）
        """
        if self.is_loaded:
            return

        print("正在加载模型和tokenizer...")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            # 根据设备选择不同的加载策略
            if self.device.type == "cuda":
                # GPU加载策略 - 使用4位量化以节省内存
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True
                )
                print("模型已以4位量化加载到GPU")
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

        except RuntimeError as e:
            if "offloaded" in str(e) or "can't move" in str(e):
                print("检测到模型卸载问题，尝试强制重新加载...")
                self._force_reload_model()
            else:
                print(f"模型加载失败: {e}")
                raise

    def _force_reload_model(self):
        """强制重新加载模型，解决卸载问题"""
        try:
            # 清理缓存
            torch.cuda.empty_cache()

            # 强制重新加载，不使用设备映射
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=None,  # 不使用自动设备映射
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=False  # 关闭低内存使用
            )

            # 手动移动到设备
            if self.device.type == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()

            self.is_loaded = True
            print("模型强制重新加载完成！")

        except Exception as e:
            print(f"强制重新加载失败: {e}")
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

    def load_training_data(self, data_path: str):
        """
        加载训练数据
        :param data_path: 数据文件路径
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载 {len(data)} 条训练数据")
            return data
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return None

    def prepare_data_for_training(self, data, test_size=0.1):
        """
        准备训练数据
        :param data: 原始数据
        :param test_size: 测试集比例
        """
        # 构建训练格式
        formatted_data = []
        for item in data:
            # 提取system和user内容
            content = item.get("system", "") + item.get("user", "")
            assistant_content = item.get("assistant", "")

            # 构建训练文本
            text = f"<|im_start|>system\n{content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
            formatted_data.append({"text": text})

        # 划分训练集和测试集
        split_idx = int(len(formatted_data) * (1 - test_size))
        train_data = formatted_data[:split_idx]
        test_data = formatted_data[split_idx:]

        print(f"训练集: {len(train_data)} 条, 测试集: {len(test_data)} 条")
        return train_data, test_data

    def setup_lora_config(self):
        """
        设置QLoRA配置（优化版本）
        """
        lora_config = LoraConfig(
            r=16,  # LoRA秩 - 提高以利用更多内存
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return lora_config

    def finetune_with_qlora(self,
                            data_path: str = "/root/-IRAS/data/data_model_train/QLora_data.txt",
                            output_dir: str = "/root/models/finetuned_model",
                            num_train_epochs: int = 5,  # 增加到5轮，利用更多内存
                            per_device_train_batch_size: int = 2,  # 增加到2，利用更多内存
                            gradient_accumulation_steps: int = 2,  # 增加到2
                            learning_rate: float = 2e-4,  # 提高学习率
                            max_seq_length: int = 1024):  # 增加到1024，利用更多内存
        """
        使用QLoRA进行模型微调（优化版本，利用翻倍内存）
        """
        print("开始QLoRA微调...")

        # 加载训练数据
        raw_data = self.load_training_data(data_path)
        if raw_data is None:
            print("无法加载训练数据，终止微调")
            return

        # 准备数据
        train_data, eval_data = self.prepare_data_for_training(raw_data)

        print(f"训练数据准备完成: {len(train_data)} 条训练, {len(eval_data)} 条验证")

        # 确保模型已加载
        if not self.is_loaded:
            self.load_model()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 检查GPU内存，利用翻倍的内存
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / (1024 ** 3)
            print(f"GPU可用内存: {free_memory_gb:.2f} GB")

            # 根据可用内存调整参数
            if free_memory_gb > 8.0:  # 如果有8GB以上可用内存
                print("检测到充足内存，使用增强参数...")
                per_device_train_batch_size = 2
                gradient_accumulation_steps = 2
                max_seq_length = 1024
            elif free_memory_gb > 4.0:  # 如果有4GB以上可用内存
                print("检测到中等内存，使用标准参数...")
                per_device_train_batch_size = 1
                gradient_accumulation_steps = 2
                max_seq_length = 512
            else:  # 内存较少
                print("内存较少，使用保守参数...")
                self._finetune_on_cpu_optimized(train_data, eval_data, output_dir, num_train_epochs, learning_rate)
                return

        try:
            # GPU训练
            self._finetune_on_gpu(train_data, eval_data, output_dir, num_train_epochs, learning_rate,
                                  max_seq_length, per_device_train_batch_size, gradient_accumulation_steps)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU内存不足，切换到CPU训练...")
                self._finetune_on_cpu_optimized(train_data, eval_data, output_dir, num_train_epochs, learning_rate)
            else:
                raise e

    def _finetune_on_gpu(self, train_data, eval_data, output_dir, num_train_epochs, learning_rate,
                         max_seq_length, per_device_train_batch_size, gradient_accumulation_steps):
        """在GPU上进行微调"""
        print("在GPU上进行微调...")
        print(f"训练参数: batch_size={per_device_train_batch_size}, "
              f"accumulation_steps={gradient_accumulation_steps}, "
              f"max_seq_length={max_seq_length}")

        # 准备模型用于训练
        self.model = prepare_model_for_kbit_training(self.model)

        # 设置LoRA配置 - 使用完整配置利用更多内存
        lora_config = LoraConfig(
            r=16,  # 提高秩
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 应用LoRA到模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 设置训练参数 - 利用更多内存
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            eval_strategy="steps",  # 启用验证
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=None,
            remove_unused_columns=True,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            fp16=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,  # 关闭以利用更多内存
        )

        def formatting_func(example):
            return example["text"]

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=Dataset.from_list(train_data),
            eval_dataset=Dataset.from_list(eval_data) if eval_data else None,
            formatting_func=formatting_func,
        )

        # 开始训练
        print("开始GPU训练...")
        trainer.train()

        # 保存模型 - 关键修复：正确保存完整模型
        print("保存模型...")

        # 合并LoRA权重到基础模型
        self.model = self.model.merge_and_unload()

        # 保存完整模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"GPU微调完成，模型保存到: {output_dir}")

        # 清理
        torch.cuda.empty_cache()

    def _finetune_on_cpu_optimized(self, train_data, eval_data, output_dir, num_train_epochs, learning_rate):
        """在CPU上进行优化的微调"""
        print("在CPU上进行优化微调...")

        # 将模型移动到CPU
        self.model = self.model.cpu()
        self.device = torch.device("cpu")

        # 准备模型用于训练
        self.model = prepare_model_for_kbit_training(self.model)

        # 设置LoRA配置
        lora_config = LoraConfig(
            r=8,  # 适当提高CPU训练的秩
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 应用LoRA到模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 设置训练参数 - CPU优化
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=len(train_data),
            eval_strategy="no",
            save_strategy="epoch",
            report_to=None,
            remove_unused_columns=True,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            use_cpu=True,
            dataloader_num_workers=0,
        )

        def formatting_func(example):
            return example["text"]

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=Dataset.from_list(train_data),
            formatting_func=formatting_func,
        )

        # 开始训练
        print("开始CPU训练...")
        trainer.train()

        # 保存模型
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"CPU微调完成，模型保存到: {output_dir}")

    def load_finetuned_model(self, model_path):
        """
        专门用于加载微调后模型的方法
        """
        print(f"加载微调模型: {model_path}")

        try:
            # 方法1：尝试标准加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        except RuntimeError as e:
            if "offloaded" in str(e):
                print("标准加载失败，尝试替代方法...")
                # 方法2：使用PeftModel加载
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    "/root/models/Qwen2.5-3B-Instruct",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )

                self.model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    device_map="auto"
                )

                # 合并权重
                self.model = self.model.merge_and_unload()
            else:
                raise e

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_loaded = True
        print("微调模型加载完成")


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    llm = LLMInferenceQ1("/root/models/Qwen2.5-3B-Instruct")

    # 显示设备信息
    print("设备信息:", llm.get_device_info())

    # 微调模型（取消注释以运行微调）
    llm.finetune_with_qlora(
        data_path="IntelligentRecruitmentAssistant/data/data_model_train/QLora_data.txt",
        output_dir="IntelligentRecruitmentAssistant/llm/finetuned_model",
        num_train_epochs=5
    )

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