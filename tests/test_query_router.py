import unittest

from rag.query_router import QueryRouter


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_route_policy_query(self):
        route = self.router.route("这个机器还在保修期吗，能不能免费修？")
        self.assertEqual(route.route, "policy")
        self.assertEqual(route.reason, "policy keywords")

    def test_route_troubleshooting_query(self):
        route = self.router.route("机器人连不上 WiFi，APP 也搜不到设备怎么办？")
        self.assertEqual(route.route, "troubleshooting")
        self.assertEqual(route.reason, "troubleshooting keywords")

    def test_route_mixed_query(self):
        route = self.router.route("机器不出水，如果是正常使用坏的还能保修吗？")
        self.assertEqual(route.route, "mixed")
        self.assertEqual(route.reason, "policy+troubleshooting keywords")

    def test_route_other_query(self):
        route = self.router.route("谢谢你，问题解决了")
        self.assertEqual(route.route, "other")
        self.assertEqual(route.reason, "no domain keywords")

    def test_route_non_warranty_damage_query(self):
        route = self.router.route("机器进水了还能保修吗？")
        self.assertEqual(route.route, "policy")

    def test_route_wifi_alias_query(self):
        route = self.router.route("APP搜不到设备，配网失败怎么处理？")
        self.assertEqual(route.route, "troubleshooting")

    def test_route_maintenance_query(self):
        route = self.router.route("拖布怎么清洗，多久换一次？")
        self.assertEqual(route.route, "maintenance")
        self.assertEqual(route.reason, "maintenance keywords")

    def test_route_maintenance_policy_mixed_query(self):
        route = self.router.route("滤网堵了怎么清理，坏了算保修吗？")
        self.assertEqual(route.route, "mixed")
        self.assertEqual(route.reason, "policy+maintenance keywords")

    def test_route_maintenance_troubleshooting_mixed_query(self):
        route = self.router.route("拖布怎么清洗，如果还是不出水要不要送修？")
        self.assertEqual(route.route, "mixed")
        self.assertEqual(route.reason, "troubleshooting+maintenance keywords")

    def test_route_policy_troubleshooting_maintenance_mixed_query(self):
        route = self.router.route("拖布不出水，怎么清洗，坏了还能保修吗？")
        self.assertEqual(route.route, "mixed")
        self.assertEqual(route.reason, "policy+troubleshooting+maintenance keywords")

    def test_route_maintenance_weak_keyword_query(self):
        route = self.router.route("日常维护要注意什么？")
        self.assertEqual(route.route, "maintenance")
        self.assertEqual(route.reason, "maintenance keywords")


if __name__ == "__main__":
    unittest.main()
