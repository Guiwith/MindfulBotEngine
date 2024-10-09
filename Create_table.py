import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    # 连接到默认的 PostgreSQL 数据库
    conn = psycopg2.connect(
        dbname="postgres",
        user="your_username",
        password="your_password",
        host="localhost",
        port="5432"
    )

    # 设置自动提交事务
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # 创建游标对象
    cur = conn.cursor()

    # 创建 mbe_db 数据库
    cur.execute('CREATE DATABASE mbe_db;')

    # 关闭游标和连接
    cur.close()
    conn.close()

    # 连接到新创建的 mbe_db 数据库
    conn = psycopg2.connect(
        dbname="mbe_db",
        user="your_username",
        password="your_password",
        host="localhost",
        port="5432"
    )

    # 创建游标对象
    cur = conn.cursor()

    # 创建 users 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS public.users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role = ANY (ARRAY['user', 'admin']))
    );
    ''')

    # 创建 history 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS public.history (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES public.users(id),
        message TEXT,
        response TEXT,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    ''')

    # 创建 knowledge_base 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS public.knowledge_base (
        id SERIAL PRIMARY KEY,
        data TEXT NOT NULL,
        embedding BYTEA,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        user_id INTEGER NOT NULL REFERENCES public.users(id)
    );
    ''')

    # 创建 teams 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS public.teams (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        is_shared BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
                ALTER TABLE team_members
DROP CONSTRAINT team_members_team_id_fkey,
ADD CONSTRAINT team_members_team_id_fkey FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE;
    ''')

    # 创建 team_members 表
    cur.execute('''
    CREATE TABLE IF NOT EXISTS public.team_members (
        team_id INTEGER REFERENCES public.teams(id),
        user_id INTEGER REFERENCES public.users(id),
        PRIMARY KEY (team_id, user_id)
    );
    ''')

    # 提交更改
    conn.commit()

    print("Database 'mbe_db' and tables created successfully.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 确保连接和游标关闭
    if cur:
        cur.close()
    if conn:
        conn.close()
