from models.encoders.qwen3_vl_embedding import Qwen3VLEmbedder
import numpy as np
import torch
from typing import List, Dict
from tqdm import tqdm

def hugging_face_demo():
    # Define a list of query texts
    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        # {"text": "Pet owner training dog outdoors near water."},
        # {"text": "Woman surfing on waves during a sunny day."},
        # {"text": "City skyline view from a high-rise building at night."}
    ]

    # Define a list of document texts and images
    documents = [
        # {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ]

    # Specify the model path
    model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"

    # Initialize the Qwen3VLEmbedder model
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    # Combine queries and documents into a single input list
    inputs = queries + documents

    # Process the inputs to get embeddings
    embeddings = model.process(inputs)

    # Compute similarity scores between query embeddings and document embeddings
    similarity_scores = (embeddings[:1] @ embeddings[1:].T)

    # Print out the similarity scores in a list format
    print(similarity_scores.tolist())

    # [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]



def use_qwen3vl_embedding_process(
        input_list: List[Dict],
        model: Qwen3VLEmbedder) -> torch.Tensor:
    """
    接收包含文本和/或图像的字典列表，并使用 Qwen3VLEmbedder 生成嵌入向量。

    参数:
        input_list: 包含输入数据的字典列表，例如 [{"text": "xxx"}, {"image": "url_or_path"}]
        model_path: 模型的本地路径或 HuggingFace 模型名称

    返回:
        embeddings: 生成的特征向量 (通常是 torch.Tensor 或 numpy array，取决于模型实现)
    """
    # 处理输入并获取嵌入向量
    embeddings = model.process(input_list)
    return embeddings


def run_train_synthetic_u_orig_text():
    # all_my_text_caps = np.load("./synthetic_u/train_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/train_text_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/train_embeds_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_valid_synthetic_u_orig_text():
    # all_my_text_caps = np.load("./synthetic_u/valid_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/valid_text_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/valid_embeds_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_test_synthetic_u_orig_text():
    # all_my_text_caps = np.load("./synthetic_u/valid_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/test_text_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/test_embeds_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_train_synthetic_u_my_text():
    # all_my_text_caps = np.load("./synthetic_u/train_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/train_text_my_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/train_embeds_my_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_valid_synthetic_u_my_text():
    # all_my_text_caps = np.load("./synthetic_u/valid_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/valid_text_my_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/valid_embeds_my_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_test_synthetic_u_my_text():
    # all_my_text_caps = np.load("./synthetic_u/valid_text_my_caps.npy", allow_pickle=True)
    all_my_text_caps = np.load("/playpen/haochenz/synthetic_u/test_text_my_caps.npy", allow_pickle=True)
    # model_name_or_path = "/Users/zhc/Downloads/Qwen3-VL-Embedding-2B"
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/synthetic_u/test_embeds_my_caps.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)


def run_sample_synthetic_u_my_text():
    all_my_text_caps = np.load("/playpen/haochenz/diffusion_prior_results/DiTDH-S-samples.npy", allow_pickle=True)
    print("DiTDH-S-samples shape: ", all_my_text_caps.shape)
    print(all_my_text_caps[0])
    # all_my_text_caps = np.load("/playpen/haochenz/diffusion_prior_results/DiTDH-XL-samples.npy", allow_pickle=True)
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    embeds = []
    for cap in tqdm(all_my_text_caps):
        input_list = [{"text": str(cap[0])}]
        embed = model.process(input_list)
        embeds.append(embed)
    embeds = torch.cat(embeds)
    torch.save(embeds, "/playpen/haochenz/diffusion_prior_results/DiTDH-S-samples_embed.pt")
    # torch.save(embeds, "/playpen/haochenz/diffusion_prior_results/DiTDH-XL-samples_embed.pt")
    print("Embeddings generated successfully!")
    print("Embedding size: ", embeds.shape)

if __name__ == '__main__':
    # run_train_synthetic_u_orig_text()
    # run_valid_synthetic_u_orig_text()
    # run_test_synthetic_u_orig_text()

    # run_train_synthetic_u_my_text()
    # run_valid_synthetic_u_my_text()
    # run_test_synthetic_u_my_text()
    run_sample_synthetic_u_my_text()


