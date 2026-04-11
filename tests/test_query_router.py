import unittest

from rag.query_router import QueryRoute, QueryRouter


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_route_policy_query(self):
        route = self.router.route("这个机器还在保修期吗，能不能免费修？")
        self.assertEqual(route.route, "policy")

    def test_route_troubleshooting_query(self):
        route = self.router.route("机器人连不上 WiFi，APP 也搜不到设备怎么办？")
        self.assertEqual(route.route, "troubleshooting")

    def test_route_mixed_query(self):
        route = self.router.route("机器不出水，如果是正常使用坏的还能保修吗？")
        self.assertEqual(route.route, "mixed")

    def test_route_other_query(self):
        route = self.router.route("谢谢你，问题解决了")
        self.assertEqual(route.route, "other")


if __name__ == "__main__":
    unittest.main()