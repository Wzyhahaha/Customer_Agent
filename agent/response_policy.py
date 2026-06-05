from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResponsePolicyResult:
    allowed: bool = True
    warnings: list[str] = field(default_factory=list)
    requires_citation: bool = False
    fallback_message: str | None = None


class ResponsePolicy:
    HIGH_RISK_KEYWORDS = [
        "免费保修", "免费维修", "免费换新", "一定保修", "肯定保修",
        "包修", "不要钱", "全额退款", "无条件", "绝对",
    ]

    WARRANTY_COMMITMENT_TEMPLATE = (
        "根据当前售后政策，{issue_type}需要进一步确认具体情况。"
        "建议您提供更多信息，或直接联系官方售后进行检测评估。"
    )

    @classmethod
    def check(cls, answer: str, confidence: float | None, risk_tags: list[str] | None = None) -> ResponsePolicyResult:
        result = ResponsePolicyResult()

        if confidence is not None and confidence < 0.3:
            result.allowed = False
            result.warnings.append("confidence_too_low")
            result.fallback_message = "抱歉，我暂时无法确定这个问题的答案。建议您联系人工客服获取更准确的帮助。"
            return result

        if risk_tags and "warranty_commitment" in risk_tags:
            result.requires_citation = True
            for keyword in cls.HIGH_RISK_KEYWORDS:
                if keyword in answer:
                    result.warnings.append(f"warranty_commitment_detected: {keyword}")
                    result.allowed = False
                    result.fallback_message = cls.WARRANTY_COMMITMENT_TEMPLATE.format(
                        issue_type="该问题"
                    )
                    break

        if risk_tags and "safety_concern" in risk_tags:
            if "不要紧" in answer or "没事" in answer or "继续用" in answer:
                result.warnings.append("safety_downplay_detected")
                result.allowed = False
                result.fallback_message = (
                    "涉及安全问题，建议您立即停止使用并联系官方售后进行专业检测。"
                )

        if confidence is not None and confidence < 0.5:
            result.warnings.append("low_confidence")

        return result
