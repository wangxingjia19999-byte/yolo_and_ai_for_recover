"""AI 智能评估模块 —— LLM 推理引擎 + 提示词工程"""

import os
import json
from typing import Optional

from .config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_TOP_K,
)
from .keypoints import EvaluationResult
from .rag_knowledge_base import get_knowledge_base


# ── 评估提示词模板 ─────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """## 角色定义
你是一位拥有20年临床经验的康复医学主任医师，擅长运动功能评估和康复方案制定。
请基于提供的患者姿态对比数据和康复医学参考知识，生成一份专业、客观的康复评估报告。

## 输出格式要求
请严格按照以下JSON格式输出评估结果：
```json
{
  "overall_score": 85,
  "grade": "良",
  "joint_analysis": [
    {
      "joint": "左膝关节",
      "deviation": 5.2,
      "normal_range": "0°-135°",
      "assessment": "活动范围正常，屈曲角度略小于标准",
      "confidence": "高"
    }
  ],
  "problems": [
    {
      "problem": "左侧髋关节屈曲受限",
      "severity": "中度",
      "possible_cause": "髋屈肌群力量不足",
      "confidence": "高"
    }
  ],
  "suggestions": [
    "建议增加髋关节屈曲肌群的力量训练",
    "训练时注意保持躯干直立，避免代偿"
  ],
  "risk_warnings": [
    "当前动作幅度下无明显二次损伤风险"
  ],
  "rag_citations": [
    "骨科术后康复指南：站立抬腿评估要点"
  ]
}
```

## 评分等级标准
- 90-100分：优 — 动作高度标准，几乎无偏差
- 75-89分：良 — 动作基本标准，轻微偏差
- 60-74分：中 — 存在明显偏差，需针对性改进
- 60分以下：差 — 动作严重不标准，存在损伤风险

## 注意事项
- 使用专业的康复医学术语
- 优先依据检索到的参考知识进行判断
- 对每个判断标注置信度（高/中/低），置信度为"低"的判断不要作为主要结论
- 标注引用的知识来源
- 语言专业但亲切，适合患者阅读
- 关节活动度正常值：肘关节0°-145°，膝关节0°-135°，肩关节0°-180°，髋关节0°-120°
"""

EVALUATION_PROMPT = """## 参考知识（来自康复医学知识库）
{rag_content}

## 本次评估数据
- 评估动作：{action_name}
- 患者动作与标准动作的关节角度偏差（单位：度）：
{angle_deviations_table}
- 关键点平均偏移量：{avg_keypoint_offset}（归一化单位）
- 动作幅度对比：
{rom_comparison_table}
- 整体相似度评分：{similarity_score}/100

请基于以上数据和参考知识，生成专业康复评估报告（JSON格式）。"""


# ── LLM 适配器 ─────────────────────────────────────────────

class LLMAdapter:
    """LLM 统一适配层，支持 OpenAI API、Ollama 本地模型等多种后端"""

    def __init__(self):
        self.provider = LLM_PROVIDER

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """
        调用LLM进行对话。

        Args:
            messages: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
            temperature: 生成温度

        Returns:
            LLM 回复文本
        """
        if self.provider == "openai":
            return self._chat_openai(messages, temperature)
        elif self.provider == "ollama":
            return self._chat_ollama(messages, temperature)
        else:
            raise ValueError(f"不支持的 LLM Provider: {self.provider}")

    def _chat_openai(self, messages: list[dict], temperature: float) -> str:
        """通过 OpenAI API 调用"""
        from openai import OpenAI

        client = OpenAI(
            api_key=LLM_API_KEY or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
            base_url=LLM_BASE_URL,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _chat_ollama(self, messages: list[dict], temperature: float) -> str:
        """通过 Ollama 本地模型调用"""
        import httpx

        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = httpx.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ── 评估引擎 ───────────────────────────────────────────────

class AIEvaluator:
    """AI 智能评估引擎"""

    def __init__(self):
        self.llm = LLMAdapter()
        try:
            self.kb = get_knowledge_base()
        except Exception:
            self.kb = None

    def _format_angle_deviations(self, deviations: dict[str, float]) -> str:
        """格式化角度偏差表"""
        lines = []
        for joint, dev in deviations.items():
            # 美化关节名
            joint_cn = joint.replace("left_", "左").replace("right_", "右") \
                .replace("elbow_angle", "肘关节").replace("knee_angle", "膝关节") \
                .replace("shoulder_angle", "肩关节").replace("hip_angle", "髋关节")
            lines.append(f"  - {joint_cn}: {dev:.1f}°")
        return "\n".join(lines) if lines else "无数据"

    def _format_rom_comparison(self, rom_data: dict[str, dict]) -> str:
        """格式化动作幅度对比表"""
        lines = []
        for joint, data in rom_data.items():
            joint_cn = joint.replace("left_", "左").replace("right_", "右") \
                .replace("elbow_angle", "肘").replace("knee_angle", "膝") \
                .replace("shoulder_angle", "肩").replace("hip_angle", "髋")
            lines.append(
                f"  - {joint_cn}: 标准范围 {data['std_range']}° vs 患者 {data['patient_range']}° "
                f"(差异 {data['difference']}°)"
            )
        return "\n".join(lines) if lines else "无数据"

    def evaluate(self, result: EvaluationResult, action_name: str) -> Optional[dict]:
        """
        执行AI评估。

        Args:
            result: 姿态对比结果
            action_name: 动作名称

        Returns:
            解析后的评估报告字典，失败返回 None
        """
        # 1. RAG 检索相关康复知识
        queries = [
            f"{action_name} 康复动作评估标准 关节活动度",
            f"康复训练 {action_name} 常见错误 改进建议",
        ]
        rag_results = []
        if self.kb is not None:
            for q in queries:
                try:
                    docs = self.kb.search(q)
                    rag_results.extend(docs)
                except Exception:
                    pass

        # 去重
        seen = set()
        unique_docs = []
        for doc in rag_results:
            if doc["content"] not in seen:
                seen.add(doc["content"])
                unique_docs.append(doc)

        rag_content = "\n\n---\n".join([
            f"【来源：{d['title']}（{d['source']}）】\n{d['content']}"
            for d in unique_docs
        ]) if unique_docs else "无相关参考知识"

        # 2. 构建提示词
        angle_table = self._format_angle_deviations(result.angle_deviations)
        rom_table = self._format_rom_comparison(result.rom_comparison)

        user_prompt = EVALUATION_PROMPT.format(
            rag_content=rag_content,
            action_name=action_name,
            angle_deviations_table=angle_table,
            avg_keypoint_offset=result.avg_keypoint_offset,
            rom_comparison_table=rom_table,
            similarity_score=result.similarity_score,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user_prompt},
        ]

        # 3. 调用 LLM
        try:
            response_text = self.llm.chat(messages, temperature=0.3)

            # 提取 JSON
            report = self._parse_json_response(response_text)

            # 附上 RAG 来源
            report["rag_sources"] = [
                f"{d['title']} ({d['source']})" for d in unique_docs
            ]
            return report

        except Exception as e:
            print(f"[AIEvaluator] LLM 调用失败: {e}")
            # 返回基础评估（不含AI分析）
            return self._fallback_evaluation(result, action_name)

    def _parse_json_response(self, text: str) -> dict:
        """从 LLM 回复中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        import re
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 提取第一个 { ... }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"raw_response": text, "parse_error": True}

    def _fallback_evaluation(self, result: EvaluationResult, action_name: str) -> dict:
        """当 LLM 不可用时的基础评估"""
        score = result.similarity_score

        if score >= 90:
            grade = "优"
        elif score >= 75:
            grade = "良"
        elif score >= 60:
            grade = "中"
        else:
            grade = "差"

        return {
            "overall_score": score,
            "grade": grade,
            "joint_analysis": [
                {
                    "joint": joint,
                    "deviation": dev,
                    "assessment": "基于角度偏差的自动评估",
                    "confidence": "中",
                }
                for joint, dev in result.angle_deviations.items()
            ],
            "problems": [],
            "suggestions": ["请连接 AI 服务以获取详细评估报告"],
            "risk_warnings": [],
            "note": "此为离线自动评估，未经 AI 模型分析",
        }
