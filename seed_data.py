"""初始化脚本 —— 构建RAG知识库并初始化数据库"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rehab_pose.database import init_database
from rehab_pose.rag_knowledge_base import get_knowledge_base


def main():
    print("=" * 60)
    print("  康复训练姿态评估系统 - 环境初始化")
    print("=" * 60)

    # 1. 初始化数据库
    print("\n[1/2] 初始化数据库...")
    init_database()

    # 2. 构建 RAG 知识库
    print("\n[2/2] 构建 RAG 康复医学知识库...")
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  ⚠️  未配置 OPENAI_API_KEY，跳过知识库向量化。")
        print("  提示：设置 OPENAI_API_KEY 环境变量后重新运行 seed_data.py 以构建知识库。")
        print("  系统将在首次 AI 评估时尝试构建知识库。")
    else:
        kb = get_knowledge_base()

    print("\n" + "=" * 60)
    print("  ✅ 初始化完成！")
    print("=" * 60)
    print("\n启动命令：")
    print("  Streamlit 前端：  uv run streamlit run app.py")
    print("  FastAPI 后端：    uv run python api.py")
    print("  API 文档：        http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    main()
