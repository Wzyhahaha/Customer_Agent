import json
import shutil
import unittest
from pathlib import Path

from scripts.build_maintenance_guides import (
    build_jsonl,
    infer_applicable_products,
    infer_frequency,
    infer_topic,
    infer_user_intents,
    parse_maintenance_text,
)


class TestBuildMaintenanceGuides(unittest.TestCase):
    def test_infer_frequency_keeps_three_to_six_months_distinct(self):
        self.assertEqual(
            infer_frequency("主刷：普通家庭3-6个月更换一次。"),
            "3-6个月",
        )

    def test_infer_topic_detects_filter_topic(self):
        self.assertEqual(
            infer_topic("耗材专项维护与更换", "滤网堵塞后需要清理。"),
            "滤网清理与更换",
        )

    def test_infer_applicable_products_uses_section_boundary(self):
        self.assertEqual(
            infer_applicable_products("通用基础维护", "每周擦拭机器人集尘口（集尘款），保证密封严实，避免漏灰。"),
            "集尘款",
        )
        self.assertEqual(
            infer_applicable_products("通用基础维护", "每月检查机器人拖地模组接口（扫拖一体），保证无灰尘、杂物，连接顺畅。"),
            "扫拖一体",
        )
        self.assertEqual(
            infer_applicable_products("耗材专项维护与更换", "污水仓滤网（扫拖一体）：每周用清水冲洗，晾干后使用，6个月更换一次，破损立即更换。"),
            "扫拖一体",
        )
        self.assertEqual(
            infer_applicable_products("耗材专项维护与更换", "集尘座滤网（集尘款）：每月清理，每6个月更换一次，破损或堵塞严重时提前更换。"),
            "集尘款",
        )
        self.assertEqual(
            infer_applicable_products("通用基础维护", "每日擦拭机身外壳。"),
            "通用",
        )

    def test_infer_frequency_covers_common_missing_patterns(self):
        self.assertEqual(
            infer_frequency("每次清扫完成，及时清理机身防撞条缝隙的毛发，防止卡扣卡顿影响回弹。"),
            "每次清扫完成后",
        )
        self.assertEqual(
            infer_frequency("每次使用前，检查水箱内无杂质、水垢，加水时用清水。"),
            "每次使用前",
        )
        self.assertEqual(
            infer_frequency("水箱密封圈：每6-12个月更换一次，若出现老化、开裂、漏水，无需等待，立即更换。"),
            "6-12个月",
        )
        self.assertEqual(
            infer_frequency("清洁液：开封后3个月内用完，密封存放于阴凉干燥处，避免阳光直射。"),
            "开封后3个月内",
        )
        self.assertEqual(
            infer_frequency("定期更新机器人固件和APP，保证系统功能正常，减少软件故障导致的操作异常。"),
            "定期",
        )

    def test_infer_topic_refines_high_frequency_keywords(self):
        self.assertEqual(
            infer_topic("通用基础维护", "每天清洁机器人充电触点，用干布擦拭机身和充电座的金属触点，避免氧化导致接触不良。"),
            "充电触点清洁与维护",
        )
        self.assertEqual(
            infer_topic("通用基础维护", "每月用干布擦拭机器人避障摄像头/传感器，避免灰尘、指纹影响识别精度。"),
            "传感器清洁与维护",
        )
        self.assertEqual(
            infer_topic("通用基础维护", "每周轻按防撞条多次，检查回弹是否正常，若卡顿及时拆解清理内部杂物。"),
            "防撞条维护",
        )
        self.assertEqual(
            infer_topic("长期存放维护", "长期存放（＞1个月），每1-2个月给机器人补电一次，将电量充至80%-90%，避免电池亏电。"),
            "长期存放准备",
        )
        self.assertEqual(
            infer_topic("环境适配维护", "木地板环境：拖地时调至低档出水量，使用干拖模式，避免地面积水。"),
            "木地板环境维护",
        )

    def test_infer_user_intents_vary_by_specific_topic(self):
        dock_intents = infer_user_intents(
            "通用基础维护",
            "充电触点清洁与维护",
            "每天清洁机器人充电触点，用干布擦拭机身和充电座的金属触点，避免氧化导致接触不良。",
        )
        sensor_intents = infer_user_intents(
            "通用基础维护",
            "传感器清洁与维护",
            "每月用干布擦拭机器人避障摄像头/传感器，避免灰尘、指纹影响识别精度。",
        )
        bumper_intents = infer_user_intents(
            "通用基础维护",
            "防撞条维护",
            "每周轻按防撞条多次，检查回弹是否正常，若卡顿及时拆解清理内部杂物。",
        )
        storage_intents = infer_user_intents(
            "长期存放维护",
            "长期存放准备",
            "长期存放（＞1个月），每1-2个月给机器人补电一次，将电量充至80%-90%，避免电池亏电。",
        )
        floor_intents = infer_user_intents(
            "环境适配维护",
            "木地板环境维护",
            "木地板环境：拖地时调至低档出水量，使用干拖模式，避免地面积水。",
        )

        self.assertTrue(any("充电座" in intent or "触点" in intent for intent in dock_intents))
        self.assertTrue(any("传感器" in intent for intent in sensor_intents))
        self.assertTrue(any("防撞条" in intent for intent in bumper_intents))
        self.assertTrue(any("长期存放" in intent for intent in storage_intents))
        self.assertTrue(any("木地板" in intent for intent in floor_intents))
        self.assertNotEqual(dock_intents, sensor_intents)
        self.assertNotEqual(storage_intents, floor_intents)

    def test_parse_full_text_retains_boundaries_and_special_fields(self):
        text = (
            "## 通用基础维护（1条）\n"
            "1. 每次清扫完成，及时清理机身防撞条缝隙的毛发、线头，防止卡扣卡顿影响回弹。\n"
            "## 长期存放维护（1条）\n"
            "1. 长期存放（＞1个月），每1-2个月给机器人补电一次，将电量充至80%-90%，避免电池亏电。\n"
        )

        records = parse_maintenance_text(text)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["section"], "通用基础维护")
        self.assertEqual(records[0]["applicable_products"], "通用")
        self.assertEqual(records[0]["topic"], "防撞条维护")
        self.assertEqual(records[0]["frequency"], "每次清扫完成后")
        self.assertEqual(records[1]["section"], "长期存放维护")
        self.assertEqual(records[1]["topic"], "长期存放准备")
        self.assertEqual(records[1]["frequency"], "1-3个月")
        self.assertTrue(any("长期存放" in intent for intent in records[1]["user_intents"]))

    def test_build_jsonl_creates_directory_and_writes_ids(self):
        text = (
            "## 通用基础维护（1条）\n"
            "1. 每日使用后，用干软布擦拭机器人机身外壳。\n"
        )

        tmp_root = Path(__file__).resolve().parent / "_tmp_build_jsonl"
        source_path = tmp_root / "input.txt"
        target_path = tmp_root / "nested" / "maintenance.jsonl"

        try:
            tmp_root.mkdir(parents=True, exist_ok=True)
            source_path.write_text(text, encoding="utf-8")

            build_jsonl(str(source_path), str(target_path))

            self.assertTrue(target_path.exists())
            lines = target_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

            payload = json.loads(lines[0])
            self.assertEqual(payload["id"], "maintenance_001")
            self.assertEqual(payload["knowledge_type"], "maintenance")
            self.assertEqual(payload["section"], "通用基础维护")
            self.assertEqual(payload["frequency"], "每次使用后")
            self.assertEqual(payload["applicable_products"], "通用")
        finally:
            if tmp_root.exists():
                shutil.rmtree(tmp_root)


if __name__ == "__main__":
    unittest.main()
