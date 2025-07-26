from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
import torch
from transformers import Trainer  # 添加这行导入
import json  # 添加这行导入


def fine_tune_llama3():
    # 配置4-bit量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 加载基础模型
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 准备QLoRA配置
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 准备模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # 加载训练数据
    dataset = load_dataset("json", data_files="data/hr_qa_dataset.json")

    # 训练配置
    training_args = TrainingArguments(
        output_dir="./llama3-recruitment",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=1000,
        logging_steps=10,
        save_steps=200,
        fp16=True
    )

    # 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=lambda data: {
            "input_ids": torch.stack([torch.tensor(d["input"]) for d in data]),
            "labels": torch.stack([torch.tensor(d["output"]) for d in data])
        }
    )
    trainer.train()
    model.save_pretrained("./llama3-recruitment-ft")


def evaluate_resume_match(job_desc: str, resume_text: str, tokenizer, model) -> dict:
    """使用微调后的模型评估简历匹配度"""
    prompt = f"""
    根据以下信息评估简历匹配度(0-100分):
    岗位要求: {job_desc}
    简历内容: {resume_text}
    
    返回JSON格式: {"score": 分数, "strengths": [优势], "weaknesses": [不足]}
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return json.loads(tokenizer.decode(outputs[0]))