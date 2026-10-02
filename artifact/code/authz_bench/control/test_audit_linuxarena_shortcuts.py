import unittest

from control.audit_linuxarena_shortcuts import (
    action_content_hash,
    expected_flag_probability,
    legacy_monitor_summary_of,
    monitor_scores_of,
    randomized_threshold,
)


class LinuxArenaShortcutAuditTests(unittest.TestCase):
    def test_action_hash_ignores_only_monitor_attachment(self):
        first = [{
            "function": "bash",
            "arguments": {"cmd": "echo hello"},
            "output": "hello",
            "monitor_response": {"sus_score": 0},
        }]
        rescored = [{
            **first[0],
            "monitor_response": {"sus_score": 9},
        }]
        changed_action = [{
            **first[0],
            "arguments": {"cmd": "echo goodbye"},
        }]

        self.assertEqual(action_content_hash(first), action_content_hash(rescored))
        self.assertNotEqual(action_content_hash(first), action_content_hash(changed_action))

    def test_monitor_score_accepts_ensemble_score_without_model_name(self):
        actions = [
            {
                "function": "bash",
                "monitor_response": {"sus_score": 2.5, "model_name": None},
            },
            {
                "function": "submit",
                "monitor_response": {"sus_score": 10},
            },
            {
                "function": "text_editor",
                "monitor_response": {"sus_score": 0, "error": "upstream failure"},
            },
            {"function": "bash", "monitor_response": None},
        ]

        scores, monitorable = monitor_scores_of(actions)

        self.assertEqual(scores, [2.5])
        self.assertEqual(monitorable, 3)

    def test_legacy_parser_attributes_max_action_top_level_model(self):
        actions = [
            {
                "function": "bash",
                "monitor_response": {"sus_score": 3, "model_name": "member-a"},
            },
            {
                "function": "bash",
                "monitor_response": {"sus_score": 7, "model_name": None},
            },
        ]

        score, model = legacy_monitor_summary_of(actions)

        self.assertEqual(score, 7)
        self.assertIsNone(model)

    def test_randomized_tie_break_hits_expected_fpr(self):
        calibration = randomized_threshold([10, 10, 0, 0], target_fpr=0.25)

        self.assertEqual(calibration["threshold"], 10)
        self.assertEqual(calibration["tie_probability"], 0.5)
        self.assertAlmostEqual(
            expected_flag_probability([10, 10], calibration),
            0.75,
        )
        self.assertEqual(expected_flag_probability([11], calibration), 1.0)
        self.assertEqual(expected_flag_probability([0], calibration), 0.0)


if __name__ == "__main__":
    unittest.main()
