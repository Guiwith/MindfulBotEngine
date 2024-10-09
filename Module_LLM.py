from config import Config
from typing import List, Union, Generator
from utils import get_db_connection
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import psycopg2.extras
import asyncio
from crawl4ai import AsyncWebCrawler
import json

# Ollama 模型类，封装对 Ollama 的调用
class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def call(self, prompt: str, stream: bool = False) -> Union[dict, Generator[dict, None, None]]:
        try:
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

# 自定义简单的 Prompt 生成逻辑，加入上下文
def generate_prompt(user_input: str, contexts: List[str]) -> str:
    context_str = "\n".join(contexts)
    return f"对话历史:\n{context_str}\n\n用户问题: {user_input}\n\n你扮演的身份：你是由亿安天下自主研发的大语言模型惑问，你使用的是70b的自研大模型。用户问题回答要求: -请基于你学习到的内容，结合用户问题回答 -请按照扮演角色的角度回答问题 -请不要提及与用户问题不相关的内容\n"

# 全局字典，用于存储每个用户的上下文
user_contexts = {}
MAX_CONTEXT_LENGTH = 5

# 使用 Config 直接初始化 OllamaLLM
ollama_llm = OllamaLLM(host=Config.OLLAMA_HOST, model=Config.LLM_MODEL)

# 包装函数用于调用 Ollama 模型
def ollama_llm_wrapper(inputs: Union[dict, str]) -> Union[dict, Generator[dict, None, None]]:
    if isinstance(inputs, str):
        full_prompt = inputs
        stream = False
    elif isinstance(inputs, dict):
        full_prompt = generate_prompt(inputs.get("user_input", ""), inputs.get("contexts", ""))
        stream = inputs.get("stream", False)
    else:
        raise ValueError(f"不支持的输入类型: {type(inputs)}")

    return ollama_llm.call(full_prompt, stream=stream)

# 生成查询关键词
def generate_query_keywords(user_message: str) -> List[str]:
    query_prompt = f"用户的问题是: {user_message}。请生成与该问题相关的简洁查询关键词，每个关键词之间用空格隔开，不要生成解释性文本或长句子。只返3到5个关键词，确保每个关键词都与用户问题高度相关。"
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
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                query = """
                    SELECT kb.data FROM knowledge_base kb
                    INNER JOIN team_members tm ON kb.user_id = tm.user_id
                    INNER JOIN teams t ON tm.team_id = t.id
                    WHERE (kb.user_id = %s OR (t.is_shared = TRUE AND tm.team_id IN 
                    (SELECT team_id FROM team_members WHERE user_id = %s)))
                    AND (""" + " OR ".join(["kb.data ILIKE %s" for _ in query_keywords]) + ")"
                cursor.execute(query, (user_id, user_id, *tuple(f'%{kw}%' for kw in query_keywords)))
                rows = cursor.fetchall()

                if not rows:
                    print("查询结果为空")
                    return []

                retrieved_data = [row['data'] for row in rows]
                if not retrieved_data:
                    return []

                user_message_vector = get_ollama_embedding(user_message)
                data_vectors = [get_ollama_embedding(data) for data in retrieved_data]
                valid_data = [(data, vector) for data, vector in zip(retrieved_data, data_vectors) if vector.size > 0]

                similarities = cosine_similarity([user_message_vector], [vector for _, vector in valid_data])[0]
                sorted_data = [data for (data, _), sim in sorted(zip(valid_data, similarities), key=lambda x: x[1], reverse=True)]

                return sorted_data[:5]
    except Exception as e:
        print(f"数据库错误: {e}")
        return []

# 爬虫类，封装 Crawl4AI 调用
class Crawler:
    def __init__(self, verbose=True):
        self.verbose = verbose

    async def fetch_content(self, url: str, js_code: list = None, css_selector: str = None) -> str:
        async with AsyncWebCrawler(verbose=self.verbose) as crawler:
            result = await crawler.arun(
                url=url,
                js_code=js_code,
                css_selector=css_selector,
                bypass_cache=True
            )
            if result.success:
                return result.extracted_content
            else:
                raise Exception("Failed to crawl the page")

    def run(self, url: str, js_code: list = None, css_selector: str = None) -> str:
        return asyncio.run(self.fetch_content(url, js_code, css_selector))

# 获取模型响应，加入上下文机制，集成爬虫
def get_model_response(user_message: str, user_id: int, stream: bool = False) -> Union[dict, Generator[dict, None, None]]:
    try:
        contexts = user_contexts.get(user_id, [])
        relevant_info = retrieve_relevant_information(user_message, user_id)

        if len(relevant_info) <= 5:
            try:
                crawler = Crawler()
                crawled_data = crawler.run(f"https://www.baidu.com/s?wd={user_message}")
                filtered_data = [item['content'] for item in json.loads(crawled_data) if isinstance(item, dict) and item.get('content', '').strip()]
                user_message_vector = get_ollama_embedding(user_message)
                data_vectors = [get_ollama_embedding(data) for data in filtered_data]
                valid_data = [(data, vector) for data, vector in zip(filtered_data, data_vectors) if vector.size > 0]

                similarities = cosine_similarity([user_message_vector], [vector for _, vector in valid_data])[0]
                sorted_data = [data for (data, _), sim in sorted(zip(valid_data, similarities), key=lambda x: x[1], reverse=True)]
                combined_content = '\n'.join(sorted_data[:15])
                contexts.append(f"这是从网络上实时爬取的精简信息：{combined_content}")
            except Exception as e:
                contexts.append(f"爬取网页时发生错误: {e}")
        
        combined_contexts = contexts + (['这是你学习到的内容：'] + relevant_info if relevant_info else [])
        response = ollama_llm_wrapper({
            "user_input": user_message,
            "contexts": combined_contexts,
            "stream": stream
        })

        user_contexts[user_id] = (user_contexts.get(user_id, []) + [f"用户: {user_message}", f"模型: {response['choices'][0]['message']['content']}"])[:MAX_CONTEXT_LENGTH]

        return (dict(choices=[{"message": {"content": line}}]) for line in response) if stream else response
    except Exception as e:
        print(f"处理错误: {e}")
        return ollama_llm._error_response(f"发生错误: {e}")