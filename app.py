from flask import Flask, request, jsonify, send_from_directory, Response,send_file
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pickle
from flask_cors import CORS
from config import Config
from functools import wraps
from Module_LLM import get_model_response
from utils import get_db_connection
from dotenv import load_dotenv, set_key
import os
import uuid
from faster_whisper import WhisperModel
import ChatTTS

app = Flask(__name__, static_folder='static', static_url_path='/')


app.config.from_object(Config)

jwt = JWTManager(app)
CORS(app)

def get_user(username):
    conn = get_db_connection()
    if conn is None:
        return None  
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        return cursor.fetchone()


def create_user(username, password, role='user'):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('INSERT INTO users (username, password, role) VALUES (%s, %s, %s)', (username, password, role))
            conn.commit()

def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user = get_jwt_identity()
        if current_user["role"] != "admin":
            return jsonify({"msg": "Admins only!"}), 403
        return fn(*args, **kwargs)
    return wrapper


@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    user = get_user(username)
    
    if user and user['password'] == password:
        access_token = create_access_token(identity={"username": username, "role": user["role"], "id": user["id"]})
        return jsonify({"access_token": access_token}), 200
    else:
        return jsonify({"msg": "Bad username or password"}), 401

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')

    if get_user(username):
        return jsonify({"msg": "User already exists"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()['count']

            role = 'admin' if user_count == 0 else 'user'
            create_user(username, password, role)

    return jsonify({"msg": "User created", "role": role}), 201

@app.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, username, role FROM users')
            users = cursor.fetchall()

    return jsonify(users)

@app.route('/users/<int:user_id>/role', methods=['PUT'])
@jwt_required()
def update_user_role(user_id):
    new_role = request.json.get('role')
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('UPDATE users SET role = %s WHERE id = %s', (new_role, user_id))
            conn.commit()

    return jsonify({"msg": "User role updated"})

@app.route('/users/<int:user_id>/password', methods=['PUT'])
@jwt_required()
def update_user_password(user_id):
    new_password = request.json.get('password')
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('UPDATE users SET password = %s WHERE id = %s', (new_password, user_id))
            conn.commit()

    return jsonify({"msg": "User password updated"})

@app.route('/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
            conn.commit()

    return jsonify({"msg": "User deleted"})

@app.route('/knowledge_base', methods=['POST'])
@jwt_required()  # 使用 jwt_required 验证登录用户
def add_knowledge_base_entry():
    current_user = get_jwt_identity()  # 获取当前登录用户的身份信息
    data = request.json.get('data')
    embedding = request.json.get('embedding')

    embedding_blob = pickle.dumps(embedding)

    # 获取当前用户的 ID（user_id），如果用户是 admin，可以选择为其他用户添加
    user_id = current_user['id']  # 假设 current_user 包含用户ID

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO knowledge_base (user_id, data, embedding, created_at) VALUES (%s, %s, %s, NOW())',
                (user_id, data, embedding_blob)
            )
            conn.commit()

    return jsonify({"msg": "Entry added to knowledge_base"}), 201


@app.route('/knowledge_base', methods=['GET'])
@jwt_required()  # 验证用户身份
def get_knowledge_base_entries():
    current_user = get_jwt_identity()  # 获取当前用户信息
    search_keyword = request.args.get('keyword', '')

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 如果是 admin，查询所有数据
            if current_user['role'] == 'admin':
                if search_keyword:
                    cursor.execute(
                        "SELECT id, user_id, data, embedding, created_at FROM knowledge_base WHERE data ILIKE %s",
                        ('%' + search_keyword + '%',)
                    )
                else:
                    cursor.execute("SELECT id, user_id, data, embedding, created_at FROM knowledge_base")
            # 如果是 user，只查询该用户自己的数据
            else:
                user_id = current_user['id']
                if search_keyword:
                    cursor.execute(
                        "SELECT id, user_id, data, embedding, created_at FROM knowledge_base WHERE user_id = %s AND data ILIKE %s",
                        (user_id, '%' + search_keyword + '%')
                    )
                else:
                    cursor.execute(
                        "SELECT id, user_id, data, embedding, created_at FROM knowledge_base WHERE user_id = %s",
                        (user_id,)
                    )

            rows = cursor.fetchall()

    entries = [
        {
            "id": row["id"],
            "user_id": row["user_id"],
            "data": row["data"],
            "embedding": pickle.loads(row["embedding"]) if row["embedding"] else None,
            "created_at": row["created_at"]
        }
        for row in rows
    ]

    return jsonify(entries), 200


@app.route('/knowledge_base/<int:entry_id>', methods=['PUT'])
@jwt_required()  # 验证用户身份
def update_knowledge_base_entry(entry_id):
    current_user = get_jwt_identity()
    data = request.json.get('data')
    embedding = request.json.get('embedding')

    embedding_blob = pickle.dumps(embedding)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 获取要更新的条目的所有者（user_id）
            cursor.execute('SELECT user_id FROM knowledge_base WHERE id = %s', (entry_id,))
            row = cursor.fetchone()

            if not row:
                return jsonify({"msg": "Entry not found"}), 404

            # 如果是 user，但条目不是该用户的，则拒绝更新
            if current_user['role'] != 'admin' and row['user_id'] != current_user['id']:
                return jsonify({"msg": "Unauthorized to update this entry"}), 403

            # 允许更新
            cursor.execute(
                'UPDATE knowledge_base SET data = %s, embedding = %s WHERE id = %s',
                (data, embedding_blob, entry_id)
            )
            conn.commit()

    return jsonify({"msg": "Entry updated"}), 200


@app.route('/knowledge_base/<int:entry_id>', methods=['DELETE'])
@jwt_required()  # 验证用户身份
def delete_knowledge_base_entry(entry_id):
    current_user = get_jwt_identity()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 获取要删除的条目的所有者（user_id）
            cursor.execute('SELECT user_id FROM knowledge_base WHERE id = %s', (entry_id,))
            row = cursor.fetchone()

            if not row:
                return jsonify({"msg": "Entry not found"}), 404

            # 如果是 user，但条目不是该用户的，则拒绝删除
            if current_user['role'] != 'admin' and row['user_id'] != current_user['id']:
                return jsonify({"msg": "Unauthorized to delete this entry"}), 403

            # 允许删除
            cursor.execute('DELETE FROM knowledge_base WHERE id = %s', (entry_id,))
            conn.commit()

    return jsonify({"msg": "Entry deleted"}), 200


@app.route('/stream-chat', methods=['POST'])
@jwt_required()
def stream_chat():
    user_message = request.json.get('message')

    def generate_response():
        for line in get_model_response(user_message, stream=True):
            if line:
                try:
                    yield f"data: {line.decode('utf-8')}\n\n"
                except UnicodeDecodeError:
                    yield f"data: [Error decoding response]\n\n"

    return Response(generate_response(), content_type='text/event-stream')

@app.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    user_message = request.json.get('message')
    current_user = get_jwt_identity()

    if 'id' not in current_user:
        return jsonify({"msg": "User ID not found"}), 400

    # 确保 user_message 是 UTF-8 编码
    user_message = user_message.encode('utf-8').decode('utf-8')

    # 传递 user_id 到 get_model_response
    user_id = current_user['id']  # 获取当前用户的ID
    model_response = get_model_response(user_message, user_id)  # 传递 user_id
    
    if model_response is None:
        return jsonify({"msg": "Error generating response from model"}), 500

    response_text = model_response.get("choices", [{}])[0].get("message", {}).get("content", "")

    print("User Message:", user_message)
    print("Model Response:", response_text)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SET client_encoding = 'UTF8';"  # 设置客户端编码
            )
            cursor.execute(
                'INSERT INTO history (user_id, message, response, timestamp) VALUES (%s, %s, %s, NOW())',
                (current_user["id"], user_message, response_text)
            )
            conn.commit()

    return jsonify({"response": response_text})

@app.route('/api/get-settings', methods=['GET'])
def get_settings():
    settings = {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_NAME': os.getenv('DB_NAME', 'mbe_db'),
        'DB_USER': os.getenv('DB_USER', 'postgres'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', '12345'),
        'OLLAMA_HOST': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        'LLM_MODEL': os.getenv('LLM_MODEL', 'lgkt/llama3-chinese-alpaca:latest'),
        'BASE_URL': os.getenv('BASE_URL', 'http://127.0.0.1:5050'),  # 新增 BASE_URL 参数
        # 前端设置
        'GLOBAL_BACKGROUND_COLOR': os.getenv('GLOBAL_BACKGROUND_COLOR', '#444444'),
        'CHAT_HISTORY_BACKGROUND_COLOR': os.getenv('CHAT_HISTORY_BACKGROUND_COLOR', '#2c3e50'),
        'USER_MESSAGE_BACKGROUND_COLOR': os.getenv('USER_MESSAGE_BACKGROUND_COLOR', 'white'),
        'USER_MESSAGE_TEXT_COLOR': os.getenv('USER_MESSAGE_TEXT_COLOR', 'black'),
        'MODEL_MESSAGE_BACKGROUND_COLOR': os.getenv('MODEL_MESSAGE_BACKGROUND_COLOR', '#7cb6c3'),
        'MODEL_MESSAGE_TEXT_COLOR': os.getenv('MODEL_MESSAGE_TEXT_COLOR', 'black'),
        'DEFAULT_LOCALE': os.getenv('DEFAULT_LOCALE', 'zh')
    }
    return jsonify(settings), 200


# 保存设置的 API
@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    data = request.json

    # 保存到 .env 文件的通用函数
    def update_env_key(key, value):
        if key in data and data[key] is not None:
            set_key('.env', key, data[key])

    # 保存后端设置
    update_env_key('DB_HOST', data.get('DB_HOST'))
    update_env_key('DB_NAME', data.get('DB_NAME'))
    update_env_key('DB_USER', data.get('DB_USER'))
    update_env_key('DB_PASSWORD', data.get('DB_PASSWORD'))
    update_env_key('OLLAMA_HOST', data.get('OLLAMA_HOST'))
    update_env_key('LLM_MODEL', data.get('LLM_MODEL'))
    update_env_key('BASE_URL', data.get('BASE_URL'))  # 新增 BASE_URL 保存逻辑

    # 保存前端设置
    update_env_key('GLOBAL_BACKGROUND_COLOR', data.get('GLOBAL_BACKGROUND_COLOR'))
    update_env_key('CHAT_HISTORY_BACKGROUND_COLOR', data.get('CHAT_HISTORY_BACKGROUND_COLOR'))
    update_env_key('USER_MESSAGE_BACKGROUND_COLOR', data.get('USER_MESSAGE_BACKGROUND_COLOR'))
    update_env_key('USER_MESSAGE_TEXT_COLOR', data.get('USER_MESSAGE_TEXT_COLOR'))
    update_env_key('MODEL_MESSAGE_BACKGROUND_COLOR', data.get('MODEL_MESSAGE_BACKGROUND_COLOR'))
    update_env_key('MODEL_MESSAGE_TEXT_COLOR', data.get('MODEL_MESSAGE_TEXT_COLOR'))
    update_env_key('DEFAULT_LOCALE', data.get('DEFAULT_LOCALE'))

    # 重新加载 .env 文件以应用新设置
    load_dotenv(override=True)

    return jsonify({"message": "Settings saved and reloaded successfully"}), 200

# TTS和STT部分
import numpy as np
from pydub import AudioSegment

AudioSegment.converter = "E:/download/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"

UPLOAD_FOLDER = 'static/uploads'

model_size = "large-v2"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

chattts_model = ChatTTS.Chat()
chattts_model.load(compile=False)

@app.route('/upload_audio', methods=['POST'])
@jwt_required()
def upload_audio():
    # 检查请求中是否有文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            # 保存文件到指定路径
            file.save(file_path)
            
            # 检查文件是否存在，并检查文件大小
            if not os.path.exists(file_path):
                return jsonify({'error': f'File {file_path} not saved correctly'}), 500
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return jsonify({'error': 'Uploaded file is empty (0 bytes)'}), 400
            
            # 打印文件大小用于调试
            print(f"File saved to {file_path}, size: {file_size} bytes")
            
            # 使用 faster-whisper 进行转录
            try:
                segments, _ = whisper_model.transcribe(file_path)
                transcription = ''.join([segment.text for segment in segments])
                print(f"Transcription result: {transcription}")  # 打印转录结果
                return jsonify({'transcription': transcription}), 200
            except Exception as e:
                print(f"Transcription error: {e}")  # 打印转录过程中的错误
                return jsonify({'error': 'Transcription failed', 'details': str(e)}), 500
        except Exception as e:
            print(f"Error saving file: {e}")  # 打印文件保存过程中的错误
            return jsonify({'error': 'File save failed', 'details': str(e)}), 500


@app.route('/tts', methods=['POST'])
@jwt_required()
def text_to_speech():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # 定义每段文本的最大字符数（340 字符）
    max_chunk_size = 340

    # 分割文本为 340 字符的片段
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    audio_segments = []

    # 对每个文本片段进行推理并生成音频
    try:
        for chunk in text_chunks:
            wavs = chattts_model.infer([chunk])  # 生成音频
            if not wavs or len(wavs) == 0:
                return jsonify({'error': f'No audio data generated for chunk: {chunk}'}), 500

            # 将生成的音频片段处理为 AudioSegment 对象
            audio_data = wavs[0]
            audio_data = np.clip(audio_data, -1, 1)
            audio_data = (audio_data * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=24000,       # 设置采样率
                sample_width=2,         # 设置样本宽度
                channels=1              # 假设是单声道音频
            )
            audio_segments.append(audio_segment)
        
        # 合并所有音频片段
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio += segment  # 逐个合并音频片段

        # 定义合并后音频文件名
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_file_path = os.path.join(UPLOAD_FOLDER, audio_filename)

        # 保存合并后的音频
        combined_audio.export(audio_file_path, format="wav")

    except Exception as e:
        return jsonify({'error': 'TTS generation failed'}), 500

    return jsonify({'audio_url': f"/uploads/{audio_filename}"}), 200


@app.route('/uploads/<filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # 如果请求的路径是静态文件，直接提供该文件
    if path.startswith('assets') or path == 'favicon.ico':
        return send_from_directory(app.static_folder, path)

    # 否则返回 index.html 让 Vue Router 处理前端路由
    return send_file(os.path.join(app.static_folder, 'index.html'))

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5050,debug=False)
