"""RAG 康复知识库模块 —— FAISS 向量检索"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from .config import (
    RAG_DOCS_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    DATA_DIR,
)


class RehabKnowledgeBase:
    """
    康复医学RAG知识库。
    使用FAISS向量检索，支持从康复医学文档中检索相关专业知识，
    以抑制LLM在康复医学领域的幻觉。
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.index_path = DATA_DIR / "faiss_index.bin"
        self.chunks_path = DATA_DIR / "rag_chunks.json"
        self._embedding_fn = None

    # ── 嵌入函数 ───────────────────────────────────────────

    def _get_embedding_fn(self):
        """获取嵌入函数"""
        if self._embedding_fn is not None:
            return self._embedding_fn

        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "未配置 OPENAI_API_KEY 环境变量。\n"
                    "请设置 OPENAI_API_KEY 后重试，或使用 Ollama 本地嵌入模型。"
                )

            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            client = OpenAI(api_key=api_key, base_url=base_url)

            def embed(texts: list[str]) -> np.ndarray:
                """调用 OpenAI embedding API"""
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
                return np.array([d.embedding for d in resp.data], dtype=np.float32)

            self._embedding_fn = embed
            return embed
        except ImportError:
            raise RuntimeError("请安装 openai 库以使用嵌入功能")

    # ── 文档处理 ───────────────────────────────────────────

    @staticmethod
    def split_text(text: str,
                   chunk_size: int = RAG_CHUNK_SIZE,
                   chunk_overlap: int = RAG_CHUNK_OVERLAP) -> list[str]:
        """
        将文本按字符数分块，支持重叠。

        Args:
            text: 原始文本
            chunk_size: 每块字符数
            chunk_overlap: 重叠字符数

        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - chunk_overlap)

        return chunks

    def add_documents(self, documents: list[dict]) -> None:
        """
        添加文档到知识库。

        Args:
            documents: [{"content": "...", "source": "...", "title": "..."}, ...]
        """
        embed_fn = self._get_embedding_fn()

        new_chunks = []
        new_metadata = []

        for doc in documents:
            text_chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(text_chunks):
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                new_chunks.append(chunk)
                new_metadata.append({
                    "source": doc.get("source", "unknown"),
                    "title": doc.get("title", "untitled"),
                    "chunk_index": i,
                    "chunk_id": chunk_hash,
                })

        if not new_chunks:
            return

        # 向量化
        embeddings = embed_fn(new_chunks)

        # 创建或更新 FAISS 索引
        if self.index is None:
            dim = embeddings.shape[1]
            # 使用内积索引（适合归一化向量的余弦相似度）
            self.index = faiss.IndexFlatIP(dim)

        # L2 归一化（余弦相似度）
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.chunks.extend(new_chunks)
        self.metadata.extend(new_metadata)

        print(f"[RAG] 已添加 {len(new_chunks)} 个文档块，知识库当前共 {self.index.ntotal} 个向量")

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        检索与查询最相关的文档片段。

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            [{"content": "...", "score": float, "source": "...", "title": "..."}, ...]
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        embed_fn = self._get_embedding_fn()
        query_embedding = embed_fn([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < RAG_SIMILARITY_THRESHOLD:
                continue
            results.append({
                "content": self.chunks[idx],
                "score": float(score),
                "source": self.metadata[idx].get("source", ""),
                "title": self.metadata[idx].get("title", ""),
            })

        return results

    def save(self) -> None:
        """持久化索引和文档块"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            print(f"[RAG] FAISS 索引已保存: {self.index_path}")

        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
            }, f, ensure_ascii=False, indent=2)
        print(f"[RAG] 文档块已保存: {self.chunks_path}")

    def load(self) -> bool:
        """从磁盘加载索引和文档块"""
        if not self.index_path.exists() or not self.chunks_path.exists():
            print("[RAG] 未找到持久化的知识库文件")
            return False

        self.index = faiss.read_index(str(self.index_path))
        print(f"[RAG] FAISS 索引已加载: {self.index_path} (共 {self.index.ntotal} 个向量)")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        print(f"[RAG] 文档块已加载: {len(self.chunks)} 个块")
        return True

    def build_default_knowledge_base(self) -> None:
        """
        构建默认康复医学知识库。
        包含康复评定标准、临床指南、训练要领等核心知识。
        """
        default_docs = [
            {
                "title": "关节活动度（ROM）正常值范围",
                "source": "康复评定学（第3版）",
                "content": """
关节活动度（Range of Motion, ROM）正常值范围：

上肢关节：
- 肩关节屈曲：0°-180°
- 肩关节伸展：0°-60°
- 肩关节外展：0°-180°
- 肘关节屈曲：0°-145°
- 肘关节伸展：145°-0°
- 前臂旋前：0°-90°
- 前臂旋后：0°-90°
- 腕关节屈曲：0°-80°
- 腕关节伸展：0°-70°

下肢关节：
- 髋关节屈曲：0°-120°
- 髋关节伸展：0°-30°
- 髋关节外展：0°-45°
- 膝关节屈曲：0°-135°
- 膝关节伸展：135°-0°
- 踝关节背屈：0°-20°
- 踝关节跖屈：0°-50°

脊柱：
- 颈椎屈曲：0°-45°
- 颈椎伸展：0°-45°
- 腰椎屈曲：0°-80°
""",
            },
            {
                "title": "Fugl-Meyer运动功能评定",
                "source": "Fugl-Meyer Assessment (FMA) Scale",
                "content": """
Fugl-Meyer运动功能评定量表（FMA）是国际上广泛使用的脑卒中后运动功能评定工具。

上肢运动功能评定（共33项，总分66分）：
- 反射活动：肱二头肌反射、肱三头肌反射
- 屈肌协同运动：肩上提、肩后缩、肩外展>90°、肩外旋、肘屈曲、前臂旋后
- 伸肌协同运动：肩内收内旋、肘伸展、前臂旋前
- 伴有协同运动的活动：手触腰椎、肩屈曲90°、前臂旋前旋后
- 分离运动：肩外展90°肘伸展、肩屈曲90°-180°肘伸展、前臂旋前旋后肘伸展
- 正常反射活动
- 腕关节稳定性与手指功能

评分标准：
0分 — 不能完成
1分 — 部分完成
2分 — 完全完成

总分分级：
<50分：严重运动功能障碍
50-84分：明显运动功能障碍
85-95分：中度运动功能障碍
96-99分：轻度运动功能障碍
""",
            },
            {
                "title": "康复训练动作要领——站立抬腿",
                "source": "骨科术后康复指南",
                "content": """
站立抬腿（Standing Hip Flexion）：

动作要领：
1. 起始姿势：双脚与肩同宽，自然站立，双手可扶椅子或墙面保持平衡。
2. 缓慢抬起一侧腿，膝关节保持伸直或微屈，髋关节屈曲至约45°-60°。
3. 保持抬腿姿势1-2秒。
4. 缓慢放下，恢复起始姿势。
5. 双腿交替进行，每组10-15次，每日3-5组。

常见错误：
- 上身倾斜代偿：患者因髋屈肌无力，抬腿时上身后仰或侧倾。正确做法应保持躯干直立。
- 抬腿过高：超出安全范围（>90°）可能导致腰部受力过大。
- 膝关节过度弯曲：将髋屈曲动作代偿为膝屈曲动作。
- 支撑腿膝关节锁死：可能导致膝关节过伸损伤。

评估要点：
- 髋关节屈曲角度是否在45°-60°目标范围内
- 躯干是否保持直立（倾斜角<5°）
- 支撑腿膝关节是否保持微屈
- 动作是否流畅，无代偿现象
""",
            },
            {
                "title": "康复训练动作要领——手臂上举",
                "source": "骨科术后康复指南",
                "content": """
手臂上举（Shoulder Flexion）：

动作要领：
1. 起始姿势：自然站立或坐姿，双臂自然下垂于体侧。
2. 缓慢向前上方举臂，肘关节保持伸直。
3. 肩关节屈曲至90°-180°（根据康复阶段调整目标角度）。
4. 保持上举姿势1-2秒。
5. 缓慢放下，恢复起始姿势。
6. 每组8-12次，每日3-5组。

常见错误：
- 耸肩代偿：肩关节活动受限时，患者容易耸肩代偿，应放松肩部。
- 躯干后仰：为达到更高角度而上身后仰。
- 肘关节弯曲：将肩屈曲代偿为肘屈曲。
- 速度不均：过快或忽快忽慢。

评估要点：
- 肩关节屈曲角度达标情况
- 肘关节是否保持伸直
- 躯干是否稳定无后仰
- 双侧肩关节活动的对称性
""",
            },
            {
                "title": "康复训练动作要领——下蹲",
                "source": "骨科术后康复指南",
                "content": """
下蹲（Squat）：

动作要领：
1. 起始姿势：双脚与肩同宽或略宽，脚尖微外八字，双手前伸或交握胸前。
2. 缓慢屈膝下蹲，保持背部挺直，膝关节方向与脚尖一致。
3. 下蹲深度：通常建议膝关节屈曲至90°（半蹲），康复后期可至120°。
4. 保持下蹲姿势1-2秒。
5. 缓慢站起，恢复起始姿势。
6. 每组10-15次，每日3组。

常见错误：
- 膝关节内扣：下蹲时膝关节向内靠拢，增加膝关节损伤风险。
- 脚跟离地：重心前移导致脚跟抬起，应保持全脚掌着地。
- 上身前倾过度：腰椎受力增大，应保持背部挺直。
- 下蹲深度不足：因恐惧或肌力不足而浅蹲。

评估要点：
- 膝关节屈曲角度
- 躯干前倾角（<30°为正常）
- 膝关节与脚尖方向一致性
- 动作对称性（双侧膝关节角度偏差<5°）
""",
            },
            {
                "title": "康复训练安全性原则",
                "source": "中国康复医学指南",
                "content": """
康复训练安全性基本原则：

1. 循序渐进原则：康复训练应从低强度、小幅度开始，逐渐增加训练量和难度。
   不可急于求成，避免过度训练导致二次损伤。

2. 无痛原则：训练过程中不应出现剧烈疼痛。轻微牵拉感和肌肉酸胀感属于正常，
   但关节刺痛、撕裂感是危险信号，应立即停止。

3. 双侧对称原则：注意双侧肢体的对称训练，避免健侧过度代偿。

4. 姿势优先原则：宁可减少训练次数和幅度，也要保证动作姿势的正确性。
   错误的姿势不仅无效，还可能导致新的损伤。

5. 个体化原则：根据患者的年龄、病情、康复阶段制定个性化训练计划。
   不同患者的训练目标和强度应有差异。

6. 风险监控：
   - 高血压患者避免头部过低动作
   - 骨质疏松患者避免高冲击动作
   - 关节置换术后严格遵循ROM限制
   - 脑卒中患者注意防跌倒

7. 停止训练的指征：
   - 训练中出现剧烈疼痛
   - 关节活动度突然下降
   - 出现异常响声或卡顿感
   - 血压异常升高（收缩压>180mmHg）
   - 明显疲劳、头晕、心慌等症状
""",
            },
            {
                "title": "Berg平衡量表",
                "source": "Berg Balance Scale (BBS)",
                "content": """
Berg平衡量表（BBS）是评估平衡功能的金标准工具，共14个项目，每项0-4分，总分56分。

项目包括：
1. 由坐到站
2. 无支撑站立
3. 无支撑坐位
4. 由站到坐
5. 转移
6. 闭目站立
7. 双脚并拢站立
8. 站立位上肢前伸
9. 站立位从地上拾物
10. 转身向后看
11. 原地转360度
12. 双脚交替踏台阶
13. 双足前后站立（无支撑）
14. 单腿站立

评分标准：
0分：需要中等或大量帮助 / 不能完成
1分：需要少量帮助 / 尝试但无法完成
2分：能完成但需要辅助 / 需要较长时间
3分：独立完成但需提示 / 需要时间较长
4分：独立安全完成 / 在规定时间内完成

总分解读：
0-20分：坐轮椅（高跌倒风险）
21-40分：辅助下步行（中等跌倒风险）
41-56分：独立步行（低跌倒风险）

在姿态评估中，单腿站立时间和站立稳定性是核心观察指标。
""",
            },
        ]

        self.add_documents(default_docs)
        self.save()
        print("[RAG] 默认康复医学知识库已构建完成")


# 全局单例
_knowledge_base: Optional[RehabKnowledgeBase] = None


def get_knowledge_base() -> RehabKnowledgeBase:
    """获取知识库单例"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = RehabKnowledgeBase()
        try:
            if not _knowledge_base.load():
                print("[RAG] 正在构建默认知识库...")
                _knowledge_base.build_default_knowledge_base()
        except Exception as e:
            print(f"[RAG] 知识库初始化失败（AI功能不可用）: {e}")
            # 返回空知识库，search() 会返回 []
    return _knowledge_base
