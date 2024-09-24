from config import Config
from typing import List, Union, Generator
from utils import get_db_connection
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ollama 模型类，封装对 Ollama 的调用
class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def call(self, prompt: str, stream: bool = False) -> Union[dict, Generator[dict, None, None]]:
        try:
            # 打印发送给模型的 POST 请求内容
            print(f"Sending POST request to model with prompt: {prompt}")
            response = self._send_request(prompt, stream)
            return self._stream_response(response) if stream else self._process_response(response)
        except requests.RequestException as e:
            return self._error_response(f"无法连接到LLM服务: {e}")
        except Exception as e:
            return self._error_response(f"处理请求时发生错误: {e}")

    def _send_request(self, prompt: str, stream: bool) -> requests.Response:
        response = requests.post(
            url=f"{self.host}/v1/chat/completions",
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": stream},
            stream=stream
        )
        response.raise_for_status()
        return response

    def _process_response(self, response: requests.Response) -> dict:
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "未识别的响应结构")
        return {"choices": [{"message": {"content": content}}]}

    def _stream_response(self, response: requests.Response) -> Generator[dict, None, None]:
        for line in response.iter_lines():
            if line:
                yield {"choices": [{"message": {"content": line.decode('utf-8')}}]}

    def _error_response(self, message: str) -> dict:
        return {"choices": [{"message": {"content": message}}]}

# 调用 Ollama 嵌入 API 获取文本向量
def get_ollama_embedding(text: str, model: str = "mxbai-embed-large") -> np.ndarray:
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        return np.array(embedding)
    except requests.RequestException as e:
        print(f"请求嵌入模型失败: {e}")
        return np.array([])

# 自定义简单的 Prompt 生成逻辑
def generate_prompt(user_input: str, contexts: str) -> str:
    return f"用户问题: {user_input}\n\n回答要求: 请基于你学习到的内容，结合用户问题回答\n你学习到的内容:\n{contexts}"

# 使用 Config 直接初始化 OllamaLLM
ollama_llm = OllamaLLM(host=Config.OLLAMA_HOST, model=Config.LLM_MODEL)

# 包装函数用于调用 Ollama 模型
def ollama_llm_wrapper(inputs: Union[dict, str]) -> Union[dict, Generator[dict, None, None]]:
    if isinstance(inputs, str):
        full_prompt = inputs
        stream = False  # 默认不使用流式传输
    elif isinstance(inputs, dict):
        full_prompt = generate_prompt(inputs.get("user_input", ""), inputs.get("contexts", ""))
        stream = inputs.get("stream", False)
    else:
        raise ValueError(f"不支持的输入类型: {type(inputs)}")

    return ollama_llm.call(full_prompt, stream=stream)

# 生成查询关键词
def generate_query_keywords(user_message: str) -> List[str]:
    query_prompt = f"用户的问题是: {user_message}。请生成与该问题相关的简洁查询关键词，每个关键词之间用空格隔开，不要生成解释性文本或长句子。只返回3到5个关键词，确保每个关键词都与用户问题高度相关。"
    response = ollama_llm_wrapper(query_prompt)
    keywords_str = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return keywords_str.split() if keywords_str else [user_message]

# 从数据库检索相关信息并按相似度排序
def retrieve_relevant_information(user_message: str, user_id: int) -> List[str]:
    query_keywords = generate_query_keywords(user_message)
    if not query_keywords:
        return []

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 查询数据库，增加对 user_id 的过滤条件
                query = """
                    SELECT data FROM knowledge_base 
                    WHERE user_id = %s AND 
                    (""" + " OR ".join(["data ILIKE %s" for _ in query_keywords]) + ")"
                
                cursor.execute(query, (user_id, *tuple(f'%{kw}%' for kw in query_keywords)))
                rows = cursor.fetchall()

                # 提取数据
                retrieved_data = [row['data'] for row in rows]
                if not retrieved_data:
                    return []

                # 用户问题向量化
                user_message_vector = get_ollama_embedding(user_message)

                # 检索数据向量化
                data_vectors = [get_ollama_embedding(data) for data in retrieved_data]
                valid_data = [(data, vector) for data, vector in zip(retrieved_data, data_vectors) if vector.size > 0]

                # 计算相似度并排序
                similarities = cosine_similarity([user_message_vector], [vector for _, vector in valid_data])[0]
                sorted_data = [data for (data, _), sim in sorted(zip(valid_data, similarities), key=lambda x: x[1], reverse=True)]

                return sorted_data[:5]  # 返回最相关的前5条数据
    except Exception as e:
        print(f"数据库错误: {e}")
        return []


# 获取模型响应
def get_model_response(user_message: str, user_id: int, stream: bool = False) -> Union[dict, Generator[dict, None, None]]:
    try:
        # 检索相关的上下文信息，传递 user_id
        contexts = retrieve_relevant_information(user_message, user_id)
        combined_contexts = "\n".join(contexts) if contexts else "无相关内容"

        # 调用模型生成回答
        response = ollama_llm_wrapper({
            "user_input": user_message,
            "contexts": combined_contexts,
            "stream": stream
        })

        return ({"choices": [{"message": {"content": line}}]} for line in response) if stream else response
    except Exception as e:
        print(f"处理错误: {e}")
        return ollama_llm._error_response(f"发生错误: {e}")
