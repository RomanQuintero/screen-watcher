import unittest

from activity import ActivityState, EventHistory
from watcher import parse_activity


class SemanticFlowTests(unittest.TestCase):
    def test_different_app_and_activity_create_new_event(self):
        history = EventHistory()
        self.assertIsNotNone(history.add_if_new(ActivityState("VS Code", "development", "Programando Screen Watcher")))
        event = history.add_if_new(ActivityState("Chrome", "web", "Consultando documentación"))
        self.assertIsNotNone(event)

    def test_same_app_and_activity_do_not_create_event(self):
        history = EventHistory()
        first = ActivityState("VS Code", "development", "Programando Screen Watcher")
        variation = ActivityState("Visual Studio Code", "development", "Revisando el código")
        self.assertIsNotNone(history.add_if_new(first))
        self.assertIsNone(history.add_if_new(variation))

    def test_json_fenced_response_is_parsed(self):
        response = '''```json
{"app":"Chrome", "activity":"web", "summary":"Consultando documentación"}
```'''
        self.assertEqual(
            parse_activity(response),
            ActivityState("Chrome", "web", "Consultando documentación"),
        )

    def test_truncated_json_keeps_known_identity(self):
        response = '{"app":"YouTube", "activity":"media", "summary":"Viendo un vídeo muy'
        self.assertEqual(parse_activity(response).identity, ("youtube", "media"))

    def test_unknown_activity_becomes_other_category(self):
        response = '{"app":"YouTube", "activity":"ver y reproducir un video", "summary":"Vídeo"}'
        self.assertEqual(parse_activity(response).activity, "other")


if __name__ == "__main__":
    unittest.main()
