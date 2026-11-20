from __future__ import annotations

from .config import CADENCE_REFERENCE, CONTACT_KNEE_REFERENCE, TORSO_LEAN_REFERENCE
from .data_types import QualitySummary, Recommendation


STRIKE_LABELS = {
    "rearfoot": "偏后掌",
    "midfoot": "中掌",
    "forefoot": "偏前掌",
    "unknown": "未判定",
}


def generate_recommendations(
    metrics: dict,
    quality: QualitySummary,
    target_cadence: tuple[float, float] | None = None,
) -> list[Recommendation]:
    target_low, target_high = target_cadence or CADENCE_REFERENCE
    advice: list[Recommendation] = []

    if quality.level in {"failed", "low"}:
        advice.append(
            Recommendation(
                kind="quality",
                title="先补一段更清楚的侧面视频",
                body=(
                    "让肩、髋、膝、踝始终在画面内，镜头约与髋部同高，"
                    "从侧面连续拍摄 8 到 15 秒。保持平常配速，避免人群遮挡。"
                ),
            )
        )
    else:
        cadence = _number(metrics.get("cadence_spm"))
        if cadence is not None and cadence < target_low - 5:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="先用节拍器缩小步幅",
                    body=(
                        f"本次约 {cadence:.0f} 步/分。保持当前配速，把节拍器设在 "
                        f"{(cadence + 5):.0f} 左右，每次只增加 5 步/分，"
                        "重点感受脚落在身体重心附近，不用刻意跨大步。"
                    ),
                )
            )
        elif cadence is not None and cadence > target_high + 12:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="先核对配速与测量结果",
                    body=(
                        f"本次约 {cadence:.0f} 步/分，高于一般慢跑参考范围。"
                        "如果实际速度并不快，先检查视频是否只有半程、双脚是否偶尔被遮挡，"
                        "再决定是否调整步频。"
                    ),
                )
            )

        strike = metrics.get("dominant_strike", "unknown")
        distribution = metrics.get("strike_distribution_pct", {})
        rearfoot_pct = _number(distribution.get("rearfoot")) or 0.0
        forefoot_pct = _number(distribution.get("forefoot")) or 0.0
        if strike == "rearfoot" and rearfoot_pct >= 60:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="降低落地时的前伸距离",
                    body=(
                        f"后掌接触约占 {rearfoot_pct:.0f}%。后跟着地本身不是错误，"
                        "但可先做 20 秒慢跑，尝试更小的步幅和更轻的落地声音；"
                        "每次只改变一个动作要点，避免突然改用前掌跑。"
                    ),
                )
            )
        elif strike == "forefoot" and forefoot_pct >= 70:
            advice.append(
                Recommendation(
                    kind="strength",
                    title="控制前掌落地后的踝部负荷",
                    body=(
                        f"前掌接触约占 {forefoot_pct:.0f}%。继续注意小腿和足底是否紧张，"
                        "训练中保留轻松跑，不要为了保持前掌姿势持续踮脚。"
                    ),
                )
            )

        contact_knee = _number(metrics.get("contact_knee_mean_deg"))
        knee_low, knee_high = CONTACT_KNEE_REFERENCE
        if contact_knee is not None and contact_knee > knee_high:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="给落地增加一点膝部缓冲",
                    body=(
                        f"触地附近膝角约 {contact_knee:.0f}°。可用 10 分钟轻松跑体会"
                        "“屈髋、落脚轻”，保持躯干稳定即可，不必主动追求深蹲式跑姿。"
                    ),
                )
            )
        elif contact_knee is not None and contact_knee < knee_low:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="复核落地帧并避免过度跨步",
                    body=(
                        f"触地附近膝角约 {contact_knee:.0f}°，可能来自跨步过大，"
                        "也可能包含着地前后的屈膝阶段。结合骨架视频确认落地点，"
                        "练习时优先缩短步幅。"
                    ),
                )
            )

        torso = _number(metrics.get("torso_lean_mean_deg"))
        torso_low, torso_high = TORSO_LEAN_REFERENCE
        if torso is not None and torso > torso_high:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="把头肩髋放回稳定轴线",
                    body=(
                        f"平均躯干前倾约 {torso:.1f}°。做 2 组 30 秒原地跑，"
                        "眼睛看向前方地面，感受肋骨下沉和核心轻微收紧，"
                        "避免弯腰代替髋部前倾。"
                    ),
                )
            )
        elif torso is not None and torso < torso_low:
            advice.append(
                Recommendation(
                    kind="strength",
                    title="练习由髋部带动身体前进",
                    body=(
                        f"平均躯干前倾约 {torso:.1f}°。加入髋铰链和臀桥练习，"
                        "让身体从踝上方形成一条自然斜线；不需要刻意后仰或大幅前压。"
                    ),
                )
            )

        if not advice:
            advice.append(
                Recommendation(
                    kind="technique",
                    title="保持当前节奏做对照训练",
                    body=(
                        "本次关键指标处于当前参考范围。保持相同配速、鞋和机位再拍一次，"
                        "比较两周内的变化，比单次追求“标准动作”更有意义。"
                    ),
                )
            )

        advice.append(
            Recommendation(
                kind="plan",
                title="下一次训练安排",
                body=(
                    "先动态热身 5 分钟。主课用轻松配速跑 4 组 3 分钟，"
                    "组间走 1 分钟，每组只关注步频或轻落地中的一个提示。"
                    "结束后做 2 组单脚平衡和 2 组慢速提踵，左右各做 8 到 12 次。"
                ),
            )
        )
        advice.append(
            Recommendation(
                kind="review",
                title="用同条件视频追踪变化",
                body=(
                    "间隔 1 到 2 周，在相同场地、相近配速和相同机位重拍。"
                    "一次只调整一个训练变量，再比较步频、膝角和着地方式的变化。"
                ),
            )
        )

    return advice[:5]


def _number(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
