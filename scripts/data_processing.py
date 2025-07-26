import pandas as pd
from tqdm import tqdm

def process_resume_data(input_csv: str, output_json: str):
    """处理简历数据为训练格式"""
    df = pd.read_csv(input_csv)
    
    processed = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed.append({
            "text": f"姓名: {row['name']}\n技能: {row['skills']}\n经验: {row['experience']}",
            "metadata": {
                "education": row["education"],
                "target_job": row["target_job"]
            }
        })
    
    pd.DataFrame(processed).to_json(output_json, orient="records", force_ascii=False)


def split_dataset(json_path: str, train_ratio: float = 0.8):
    """分割数据集为训练集和验证集"""
    import json
    from sklearn.model_selection import train_test_split
    
    with open(json_path) as f:
        data = json.load(f)
    
    train, val = train_test_split(data, train_size=train_ratio)
    
    with open("train_data.json", "w") as f:
        json.dump(train, f, ensure_ascii=False)
    
    with open("val_data.json", "w") as f:
        json.dump(val, f, ensure_ascii=False)