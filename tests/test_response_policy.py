from agent.response_policy import ResponsePolicy


class TestResponsePolicy:
    def test_normal_answer_passes(self):
        result = ResponsePolicy.check("请检查滤网是否堵塞。", 0.8)
        assert result.allowed is True

    def test_too_low_confidence_blocks(self):
        result = ResponsePolicy.check("可能是电池问题。", 0.2)
        assert result.allowed is False
        assert "confidence_too_low" in result.warnings

    def test_warranty_commitment_blocked(self):
        result = ResponsePolicy.check(
            "这个情况肯定保修，免费维修不要钱。", 0.8, ["warranty_commitment"]
        )
        assert result.allowed is False
        assert any("warranty_commitment_detected" in w for w in result.warnings)

    def test_safety_concern_blocks_downplay(self):
        result = ResponsePolicy.check(
            "冒烟了不要紧，继续用就行。", 0.9, ["safety_concern"]
        )
        assert result.allowed is False
        assert "safety_downplay_detected" in result.warnings

    def test_warranty_tag_requires_citation(self):
        result = ResponsePolicy.check("建议联系售后检测。", 0.8, ["warranty_commitment"])
        assert result.requires_citation is True

    def test_low_confidence_warns(self):
        result = ResponsePolicy.check("可能是这个问题。", 0.4)
        assert "low_confidence" in result.warnings
