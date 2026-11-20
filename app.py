from __future__ import annotations

import html
import io
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from posecoach.config import (
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_DETECTION_CONFIDENCE,
    DEFAULT_MAX_DURATION_S,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MODEL_PATH,
    RUNS_DIR,
    SUPPORTED_VIDEO_TYPES,
)
from posecoach.data_types import AnalysisResult
from posecoach.model import model_is_ready
from posecoach.pipeline import AnalysisError, analyze_video
from posecoach.recommendations import STRIKE_LABELS


st.set_page_config(
    page_title="智姿护航 · AI 跑姿分析",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    :root {
        --ink: #19312e;
        --muted: #60736e;
        --paper: #f6f7f2;
        --mint: #dff4e8;
        --green: #087f6b;
        --deep-green: #123d38;
        --lime: #c8f46b;
        --line: rgba(25, 49, 46, 0.12);
    }
    html, body, [class*="css"] {
        font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 84% 8%, rgba(200, 244, 107, 0.18), transparent 26rem),
            linear-gradient(180deg, #fbfcf7 0%, var(--paper) 55%, #eef3ed 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #edf3ed;
        border-right: 1px solid var(--line);
    }
    .block-container {
        max-width: 1260px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        padding: 3.1rem 3.2rem;
        color: #f7fff9;
        background:
            radial-gradient(circle at 86% 18%, rgba(200, 244, 107, 0.32), transparent 17rem),
            linear-gradient(125deg, #102d2b 0%, #164943 62%, #0b6f60 100%);
        box-shadow: 0 24px 70px rgba(18, 61, 56, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 50%;
        right: -70px;
        bottom: -210px;
        box-shadow: 0 0 0 34px rgba(255,255,255,0.035), 0 0 0 70px rgba(255,255,255,0.025);
    }
    .hero-kicker {
        color: var(--lime);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero h1 {
        max-width: 780px;
        margin: 0;
        color: #ffffff;
        font-size: clamp(2.2rem, 3.6vw, 3.9rem);
        line-height: 1.06;
        letter-spacing: -0.045em;
    }
    .hero p {
        max-width: 720px;
        color: rgba(247,255,249,0.78);
        font-size: 1.04rem;
        line-height: 1.8;
        margin: 1.3rem 0 0;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.5rem;
    }
    .hero-badge {
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.78rem;
        color: #ecfff6;
    }
    .section-kicker {
        color: var(--green);
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0 0 0.65rem;
    }
    .guide-card {
        height: 100%;
        background: rgba(255,255,255,0.68);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.5rem 1.55rem;
        box-shadow: 0 12px 34px rgba(18,61,56,0.055);
    }
    .guide-number {
        display: inline-grid;
        place-items: center;
        width: 28px;
        height: 28px;
        margin-right: 0.55rem;
        border-radius: 9px;
        color: #0d4f45;
        background: var(--mint);
        font-size: 0.78rem;
        font-weight: 800;
    }
    .guide-title {
        color: var(--ink);
        font-size: 1.04rem;
        font-weight: 750;
        margin-bottom: 0.85rem;
    }
    .guide-item {
        display: flex;
        gap: 0.6rem;
        color: var(--muted);
        font-size: 0.91rem;
        line-height: 1.65;
        margin: 0.68rem 0;
    }
    .guide-dot {
        flex: 0 0 auto;
        width: 7px;
        height: 7px;
        margin-top: 0.53rem;
        border-radius: 50%;
        background: var(--green);
    }
    .kpi-card {
        min-height: 145px;
        height: 100%;
        padding: 1.2rem 1.25rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        box-shadow: 0 10px 28px rgba(18,61,56,0.05);
    }
    .kpi-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .kpi-value {
        color: var(--ink);
        font-size: 2.22rem;
        font-weight: 850;
        letter-spacing: -0.05em;
        line-height: 1.15;
        margin-top: 0.6rem;
    }
    .kpi-unit {
        margin-left: 0.18rem;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0;
    }
    .kpi-note {
        color: #73857f;
        font-size: 0.76rem;
        line-height: 1.5;
        margin-top: 0.58rem;
    }
    .quality-panel {
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        margin: 0.7rem 0 1.25rem;
        border: 1px solid rgba(8,127,107,0.16);
        background: rgba(223,244,232,0.72);
    }
    .quality-panel.warning {
        border-color: rgba(183,121,21,0.22);
        background: rgba(255,244,219,0.78);
    }
    .quality-panel.danger {
        border-color: rgba(185,65,65,0.22);
        background: rgba(255,235,235,0.8);
    }
    .quality-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .quality-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
    }
    .quality-score {
        color: var(--green);
        font-size: 0.8rem;
        font-weight: 800;
    }
    .quality-copy {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.65;
        margin-top: 0.48rem;
    }
    .advice-card {
        height: 100%;
        border-radius: 22px;
        padding: 1.35rem 1.4rem;
        background: rgba(255,255,255,0.8);
        border: 1px solid var(--line);
        box-shadow: 0 10px 30px rgba(18,61,56,0.045);
        margin-bottom: 0.85rem;
    }
    .advice-kind {
        display: inline-block;
        color: #0b6759;
        background: var(--mint);
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.7rem;
    }
    .advice-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.48rem;
    }
    .advice-body {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.75;
    }
    .upload-note {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .footer-note {
        color: #7f8e89;
        font-size: 0.76rem;
        line-height: 1.7;
        border-top: 1px solid var(--line);
        margin-top: 2.2rem;
        padding-top: 1rem;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.65);
        border-radius: 22px;
        padding: 0.65rem;
        border: 1px solid var(--line);
    }
    div.stButton > button {
        min-height: 3rem;
        border-radius: 14px;
        border: 1px solid #087f6b;
        background: #087f6b;
        color: white;
        font-weight: 800;
        letter-spacing: 0.02em;
        box-shadow: 0 9px 22px rgba(8,127,107,0.18);
    }
    div.stButton > button:hover {
        border-color: #075f52;
        background: #075f52;
        color: white;
    }
    div.stDownloadButton > button {
        border-radius: 12px;
        border-color: rgba(8,127,107,0.35);
    }
    div[data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 999px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 1rem; }
        .hero { padding: 2rem 1.35rem; border-radius: 22px; }
        .hero h1 { font-size: 2.3rem; }
        .kpi-card { min-height: 126px; }
    }
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def main() -> None:
    _render_hero()
    settings = _render_sidebar()
    uploaded = st.file_uploader(
        "上传跑步视频",
        type=list(SUPPORTED_VIDEO_TYPES),
        accept_multiple_files=False,
        help="建议 8–15 秒、全身侧拍、画面稳定、单人出镜。",
    )

    if uploaded is None:
        _render_start_guide()
        _render_footer()
        return

    file_key = f"{uploaded.name}:{uploaded.size}"
    _render_upload_preview(uploaded)
    analyze_clicked = st.button(
        "开始 AI 跑姿分析",
        type="primary",
        width="stretch",
    )

    if analyze_clicked:
        _run_analysis(uploaded, file_key, settings)

    result = st.session_state.get("analysis_result")
    result_file_key = st.session_state.get("analysis_file_key")
    if isinstance(result, AnalysisResult) and result_file_key == file_key:
        _render_result(result)

    _render_footer()


def _render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">AI Running Form Lab · 智姿护航</div>
            <h1>看懂每一次落地，<br>把跑姿变成可练习的线索。</h1>
            <p>
                上传一段侧面跑步视频，系统会逐帧识别人体骨骼，
                计算步频、膝角、髋角、躯干前倾与着地方式，
                再给出针对本次视频的训练建议。
            </p>
            <div class="hero-badges">
                <span class="hero-badge">33 点姿态识别</span>
                <span class="hero-badge">骨架视频回放</span>
                <span class="hero-badge">指标时间序列</span>
                <span class="hero-badge">可下载分析报告</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 分析设置")
        st.caption("比赛演示建议直接使用默认值。")
        analysis_fps = st.select_slider(
            "姿态采样帧率",
            options=[8, 10, 12, 16, 20],
            value=DEFAULT_ANALYSIS_FPS,
            help="越高越流畅，但普通笔记本处理会更慢。",
        )
        max_duration = st.slider(
            "最长分析时长",
            min_value=15,
            max_value=180,
            value=DEFAULT_MAX_DURATION_S,
            step=15,
            format="%d 秒",
        )
        max_width = st.select_slider(
            "输入画面宽度",
            options=[960, 1280, 1600],
            value=DEFAULT_MAX_WIDTH,
            format_func=lambda value: f"{value} px",
        )
        target_cadence = st.slider(
            "个人目标步频",
            min_value=150,
            max_value=200,
            value=(165, 185),
            step=5,
            help="只用于生成改进建议，不会改变测量结果。",
        )
        with st.expander("高级参数"):
            confidence = st.slider(
                "关键点检测置信度",
                min_value=0.30,
                max_value=0.80,
                value=DEFAULT_DETECTION_CONFIDENCE,
                step=0.05,
            )

        st.divider()
        if model_is_ready(DEFAULT_MODEL_PATH):
            st.success("姿态模型已就绪", icon="✅")
        else:
            st.warning("首次分析将自动下载约 6 MB 的官方姿态模型。")

        st.markdown("### 拍摄建议")
        st.caption(
            "镜头与髋部同高 · 全身入镜 · 侧面或斜侧 15° · "
            "连续拍摄 8–15 秒 · 保持平常配速"
        )
        st.caption(
            "手机建议使用 1080p、30 fps 或 60 fps，避免慢动作与数字防抖裁切。"
        )

    return {
        "analysis_fps": analysis_fps,
        "max_duration": max_duration,
        "max_width": max_width,
        "target_cadence": (float(target_cadence[0]), float(target_cadence[1])),
        "confidence": float(confidence),
    }


def _render_start_guide() -> None:
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.markdown(
            """
            <div class="section-kicker">Step 01</div>
            <div class="section-title">先拍一段能看清全身的跑步视频</div>
            <div class="guide-card">
                <div class="guide-title">这条分析链路会做什么</div>
                <div class="guide-item"><span class="guide-dot"></span><span>MediaPipe 逐帧识别 33 个身体关键点。</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>OpenCV 将骨架与膝角实时叠回视频。</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>算法从双脚垂直运动轨迹估计触地与步频。</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>规则引擎结合指标与画面质量生成训练建议。</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>结果可下载为骨架视频、CSV 指标和 JSON 报告。</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="section-kicker">Step 02</div>
            <div class="section-title">拍摄检查清单</div>
            <div class="guide-card">
                <div class="guide-item"><span class="guide-number">1</span><span><b>背景干净：</b>画面中只保留跑步者，避免路人重叠。</span></div>
                <div class="guide-item"><span class="guide-number">2</span><span><b>机位固定：</b>手机垂直于跑动方向，镜头与髋部同高。</span></div>
                <div class="guide-item"><span class="guide-number">3</span><span><b>全身入镜：</b>头顶到脚尖都要在画面内，左右各留一点空间。</span></div>
                <div class="guide-item"><span class="guide-number">4</span><span><b>速度自然：</b>按日常配速跑，不要为了拍摄刻意改步频。</span></div>
                <div class="guide-item"><span class="guide-number">5</span><span><b>时长够用：</b>至少 4 秒，推荐 8–15 秒，包含 6 次以上触地。</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_upload_preview(uploaded) -> None:
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.video(uploaded.getvalue())
    with right:
        size_mb = uploaded.size / (1024 * 1024)
        st.markdown(
            f"""
            <div class="guide-card">
                <div class="section-kicker">Ready Check</div>
                <div class="section-title" style="font-size:1.25rem;">{html.escape(uploaded.name)}</div>
                <div class="guide-item"><span class="guide-dot"></span><span>文件大小：{size_mb:.1f} MB</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>建议分析时长：8–15 秒</span></div>
                <div class="guide-item"><span class="guide-dot"></span><span>首轮会逐帧识别并生成骨架回放</span></div>
                <div class="upload-note">分析只在本机进程内完成。视频、结果和下载文件默认保存在项目的 data/runs 目录。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_analysis(uploaded, file_key: str, settings: dict) -> None:
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {f".{item}" for item in SUPPORTED_VIDEO_TYPES}:
        suffix = ".mp4"
    input_path = run_dir / f"input{suffix}"
    input_path.write_bytes(uploaded.getvalue())

    progress = st.progress(0.0, text="正在准备分析")
    try:
        result = analyze_video(
            input_path,
            run_dir,
            analysis_fps=settings["analysis_fps"],
            detection_confidence=settings["confidence"],
            max_duration_s=settings["max_duration"],
            max_width=settings["max_width"],
            target_cadence=settings["target_cadence"],
            progress_callback=lambda ratio, message: progress.progress(
                ratio, text=message
            ),
        )
        st.session_state["analysis_result"] = result
        st.session_state["analysis_file_key"] = file_key
        progress.empty()
        st.success("分析完成，结果已生成。")
    except AnalysisError as exc:
        progress.empty()
        st.error(str(exc))
    except Exception as exc:
        progress.empty()
        st.error(f"未预期的分析错误：{exc}")


def _render_result(result: AnalysisResult) -> None:
    metrics = result.metrics
    st.divider()
    st.markdown(
        """
        <div class="section-kicker">Analysis Result</div>
        <div class="section-title">本次跑姿复盘</div>
        """,
        unsafe_allow_html=True,
    )
    quality_class = (
        "warning"
        if result.quality.level == "medium"
        else "danger"
        if result.quality.level in {"low", "failed"}
        else ""
    )
    st.markdown(
        f"""
        <div class="quality-panel {quality_class}">
            <div class="quality-head">
                <div class="quality-title">{html.escape(result.quality.label)}</div>
                <div class="quality-score">画面质量 {result.quality.score}/100</div>
            </div>
            <div class="quality-copy">{html.escape(result.quality.message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cadence = metrics.get("cadence_spm")
    contact_knee = metrics.get("contact_knee_mean_deg")
    torso = metrics.get("torso_lean_mean_deg")
    strike = STRIKE_LABELS.get(metrics.get("dominant_strike"), "未判定")
    columns = st.columns(4, gap="medium")
    with columns[0]:
        _metric_card(
            "步频",
            _format_number(cadence, 0),
            "步/分",
            metrics.get("cadence_source") or "需要更多有效触地",
        )
    with columns[1]:
        _metric_card(
            "触地膝角",
            _format_number(contact_knee, 0),
            "°",
            "数值含视频二维投影误差",
        )
    with columns[2]:
        _metric_card(
            "平均躯干前倾",
            _format_number(torso, 1),
            "°",
            "相对于画面竖直方向",
        )
    with columns[3]:
        _metric_card(
            "主要着地方式",
            strike,
            "",
            f"有效触地 {metrics.get('contact_count', 0)} 次",
        )

    tabs = st.tabs(["骨架回放", "指标曲线", "训练建议"])
    with tabs[0]:
        _render_video_tab(result)
    with tabs[1]:
        _render_metrics_tab(result)
    with tabs[2]:
        _render_advice_tab(result)


def _render_video_tab(result: AnalysisResult) -> None:
    left, right = st.columns([1.35, 0.65], gap="large")
    with left:
        video_mime = "video/webm" if result.output_video_path.suffix == ".webm" else "video/mp4"
        st.video(result.output_video_path.read_bytes(), format=video_mime)
    with right:
        metadata = result.metadata
        st.markdown("#### 分析信息")
        st.write(
            f"**{metadata.analyzed_duration_s:.1f} 秒** · "
            f"{metadata.analyzed_frame_count} 个分析帧"
        )
        st.caption(
            f"源视频 {metadata.source_fps:.1f} fps，姿态采样 "
            f"{metadata.analysis_fps:.1f} fps，输出 {metadata.width}×{metadata.height}。"
        )
        st.progress(
            min(metrics_percent(result.metrics.get("pose_coverage_pct")), 1.0),
            text=f"人体识别覆盖率 {result.metrics.get('pose_coverage_pct', 0):.1f}%",
        )
        if metadata.truncated:
            st.warning("视频超过设置的最长分析时长，本次只分析了前段画面。")

        report_bytes = json.dumps(
            result.to_report_dict(),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        video_bytes = result.output_video_path.read_bytes()
        st.download_button(
            "下载骨架视频",
            data=video_bytes,
            file_name=f"pose_overlay{result.output_video_path.suffix}",
            mime=video_mime,
            width="stretch",
            on_click="ignore",
        )
        st.download_button(
            "下载 JSON 分析报告",
            data=report_bytes,
            file_name=f"pose_report_{result.run_id}.json",
            mime="application/json",
            width="stretch",
            on_click="ignore",
        )


def _render_metrics_tab(result: AnalysisResult) -> None:
    metrics = result.metrics
    rows = [
        ["步频", _format_number(metrics.get("cadence_spm"), 0), "步/分"],
        ["触地膝角均值", _format_number(metrics.get("contact_knee_mean_deg"), 1), "°"],
        ["膝关节角度 P10–P90", _range_text(metrics, "knee_p10_deg", "knee_p90_deg"), "°"],
        ["左膝均值", _format_number(metrics.get("left_knee_mean_deg"), 1), "°"],
        ["右膝均值", _format_number(metrics.get("right_knee_mean_deg"), 1), "°"],
        ["髋关节角度均值", _format_number(metrics.get("hip_mean_deg"), 1), "°"],
        ["躯干前倾 P10–P90", _range_text(metrics, "torso_lean_p10_deg", "torso_lean_p90_deg"), "°"],
        ["人体识别覆盖率", _format_number(metrics.get("pose_coverage_pct"), 1), "%"],
    ]
    dataframe = pd.DataFrame(rows, columns=["指标", "数值", "单位"])

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.markdown("#### 指标汇总")
        st.dataframe(dataframe, hide_index=True, width="stretch", height=330)
        distribution = metrics.get("strike_distribution_pct") or {}
        if distribution:
            strike_series = pd.Series(
                {
                    STRIKE_LABELS.get(key, key): value
                    for key, value in distribution.items()
                },
                name="占比(%)",
            )
            st.markdown("#### 着地方式分布")
            st.bar_chart(strike_series, color="#087f6b", height=220)
    with right:
        st.markdown("#### 关节角度时间序列")
        series = pd.DataFrame(result.time_series_rows())
        chart_columns = [
            "左膝角度(°)",
            "右膝角度(°)",
            "左髋角度(°)",
            "右髋角度(°)",
            "躯干前倾(°)",
        ]
        chart = series.set_index("时间(s)")[chart_columns]
        st.line_chart(
            chart,
            color=["#37a8d8", "#e28b32", "#8bc6ad", "#c6a66b", "#516e66"],
            height=330,
        )
        st.caption(
            "虚线尖峰或局部缺失通常来自遮挡、关键点抖动或人物离开画面，"
            "应结合骨架回放复核。"
        )

    csv_buffer = io.StringIO()
    series.to_csv(csv_buffer, index=False)
    st.download_button(
        "下载逐帧指标 CSV",
        data=("\ufeff" + csv_buffer.getvalue()).encode("utf-8"),
        file_name=f"pose_metrics_{result.run_id}.csv",
        mime="text/csv",
        on_click="ignore",
    )


def _render_advice_tab(result: AnalysisResult) -> None:
    left, right = st.columns(2, gap="large")
    for index, advice in enumerate(result.recommendations):
        target = left if index % 2 == 0 else right
        kind_label = {
            "quality": "拍摄复核",
            "technique": "动作提示",
            "strength": "力量补充",
            "plan": "训练安排",
            "review": "效果追踪",
        }.get(advice.kind, "训练提示")
        with target:
            st.markdown(
                f"""
                <div class="advice-card">
                    <div class="advice-kind">{kind_label}</div>
                    <div class="advice-title">{html.escape(advice.title)}</div>
                    <div class="advice-body">{html.escape(advice.body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "这是基于单相机二维关键点的运动教学参考，不是医疗诊断。"
        "若跑步中出现持续疼痛、肿胀或明显不对称，请停止训练并咨询运动医学或康复专业人士。",
        icon="ℹ️",
    )


def _metric_card(label: str, value: str, unit: str, note: str) -> None:
    unit_html = f'<span class="kpi-unit">{html.escape(unit)}</span>' if unit else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}{unit_html}</div>
            <div class="kpi-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value, digits: int) -> str:
    try:
        return "—" if value is None else f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _range_text(metrics: dict, low_key: str, high_key: str) -> str:
    low = metrics.get(low_key)
    high = metrics.get(high_key)
    if low is None or high is None:
        return "—"
    return f"{float(low):.1f}–{float(high):.1f}"


def metrics_percent(percent) -> float:
    try:
        return max(0.0, min(float(percent) / 100.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _render_footer() -> None:
    st.markdown(
        """
        <div class="footer-note">
            智姿护航 · 计算机视觉跑步损伤风险评估与科学训练平台原型<br>
            OpenCV 负责视频读写与骨架绘制，MediaPipe Pose Landmarker 负责 33 点识别，
            Streamlit 提供上传、分析与报告展示界面。测量结果仅用于运动教学和产品演示。
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
