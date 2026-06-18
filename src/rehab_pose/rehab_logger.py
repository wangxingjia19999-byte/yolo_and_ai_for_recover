"""康复日志系统 —— 训练记录、趋势可视化、报告管理"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .database import get_rehab_logs, get_score_trend, get_rehab_log


def generate_score_trend_chart(patient_id: str, action_name: str = None):
    """
    生成评分趋势折线图。

    以时间为横轴、评分为纵轴，展示患者历次训练的评分变化趋势。

    Returns:
        plotly Figure 对象
    """
    data = get_score_trend(patient_id, action_name)

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无训练记录",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    df = pd.DataFrame(data)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # 按动作类型分组着色
    fig = px.line(
        df,
        x="created_at",
        y="similarity_score",
        color="action_name",
        markers=True,
        title=f"患者 {patient_id} 评分趋势",
        labels={
            "created_at": "训练日期",
            "similarity_score": "相似度评分",
            "action_name": "训练动作",
        },
    )

    # 添加得分等级背景色带
    fig.add_hrect(y0=90, y1=100, line_width=0, fillcolor="green", opacity=0.1,
                  annotation_text="优", annotation_position="top right")
    fig.add_hrect(y0=75, y1=90, line_width=0, fillcolor="blue", opacity=0.1,
                  annotation_text="良", annotation_position="top right")
    fig.add_hrect(y0=60, y1=75, line_width=0, fillcolor="orange", opacity=0.1,
                  annotation_text="中", annotation_position="top right")
    fig.add_hrect(y0=0, y1=60, line_width=0, fillcolor="red", opacity=0.1,
                  annotation_text="差", annotation_position="top right")

    fig.update_layout(
        yaxis_range=[0, 100],
        height=400,
        hovermode="x unified",
    )

    return fig


def generate_joint_radar_chart(current_angles: dict[str, float],
                               historical_best: dict[str, float] = None):
    """
    生成关节改善雷达图。

    以各关键关节为维度，展示当前评估与历史最佳之间的对比。

    Args:
        current_angles: 当前关节角度字典
        historical_best: 历史最佳角度字典

    Returns:
        plotly Figure 对象
    """
    categories = list(current_angles.keys())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[current_angles.get(c, 0) for c in categories],
        theta=categories,
        fill="toself",
        name="当前评估",
        line_color="blue",
    ))

    if historical_best:
        fig.add_trace(go.Scatterpolar(
            r=[historical_best.get(c, 0) for c in categories],
            theta=categories,
            fill="toself",
            name="历史最佳",
            line_color="green",
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="关节功能改善雷达图",
        height=450,
    )

    return fig


def generate_training_heatmap(patient_id: str):
    """
    生成训练频次热力图。

    以日历热力图形式展示训练频率，颜色深浅表示训练密集程度。

    Returns:
        plotly Figure 对象
    """
    data = get_rehab_logs(patient_id, limit=200)

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无训练记录",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    daily_counts = df.groupby("date").size().reset_index(name="count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    # 确保日期连续
    if len(daily_counts) > 1:
        date_range = pd.date_range(
            start=daily_counts["date"].min(),
            end=max(daily_counts["date"].max(), pd.Timestamp.today()),
        )
        daily_counts = daily_counts.set_index("date")
        daily_counts = daily_counts.reindex(date_range, fill_value=0).reset_index()
        daily_counts.columns = ["date", "count"]

    daily_counts["weekday"] = daily_counts["date"].dt.dayofweek
    daily_counts["week"] = daily_counts["date"].dt.isocalendar().week.astype(int)

    fig = go.Figure(data=go.Heatmap(
        z=daily_counts["count"],
        x=daily_counts["date"],
        y=daily_counts["weekday"],
        colorscale="YlOrRd",
        hoverongaps=False,
        colorbar=dict(title="训练次数"),
    ))

    fig.update_layout(
        title="训练频次热力图",
        height=200,
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
        ),
    )

    return fig


def format_report_for_display(report: dict) -> str:
    """
    将评估报告格式化为 Markdown 字符串，适合 Streamlit 展示。

    Args:
        report: AI 评估报告字典

    Returns:
        格式化后的 Markdown 文本
    """
    lines = []

    score = report.get("overall_score", "N/A")
    grade = report.get("grade", "N/A")

    # 评分等级颜色
    grade_emoji = {"优": "🟢", "良": "🔵", "中": "🟡", "差": "🔴"}.get(grade, "")

    lines.append(f"## {grade_emoji} 总体评分：{score}/100（{grade}）")
    lines.append("")

    # 关节分析
    if report.get("joint_analysis"):
        lines.append("### 🔬 各关节详细分析")
        lines.append("")
        lines.append("| 关节 | 偏差 | 正常范围 | 评估 | 置信度 |")
        lines.append("|------|------|----------|------|--------|")
        for item in report["joint_analysis"]:
            dev = item.get("deviation", "N/A")
            lines.append(
                f"| {item['joint']} | {dev}° | {item.get('normal_range', '-')} "
                f"| {item['assessment']} | {item.get('confidence', '-')} |"
            )
        lines.append("")

    # 问题诊断
    if report.get("problems"):
        lines.append("### ⚠️ 问题诊断")
        lines.append("")
        for p in report["problems"]:
            sev = p.get("severity", "")
            sev_emoji = {"轻度": "⚪", "中度": "🟡", "重度": "🔴"}.get(sev, "")
            lines.append(f"- **{sev_emoji} {p['problem']}**（{sev}）")
            if p.get("possible_cause"):
                lines.append(f"  - 可能原因：{p['possible_cause']}")
            lines.append(f"  - 置信度：{p.get('confidence', 'N/A')}")
        lines.append("")

    # 改进建议
    if report.get("suggestions"):
        lines.append("### 💡 改进建议")
        lines.append("")
        for s in report["suggestions"]:
            lines.append(f"- {s}")
        lines.append("")

    # 风险提示
    if report.get("risk_warnings"):
        lines.append("### 🚨 风险提示")
        lines.append("")
        for w in report["risk_warnings"]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    # 知识来源
    if report.get("rag_sources"):
        lines.append("### 📚 参考知识来源")
        lines.append("")
        for src in report["rag_sources"]:
            lines.append(f"- {src}")
        lines.append("")

    return "\n".join(lines)
