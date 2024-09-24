import psycopg2
import psycopg2.extras
from config import Config

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None
