#!/usr/bin/env python3
"""Tests for py/gen_scenarios.py — scenario-seeded dialogue generator."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import gen_scenarios as gs


class ParsePairLine(unittest.TestCase):

    def test_basic_pipe_split(self):
        q, r = gs.parse_pair_line("HELLO THERE|HEY! HOW CAN I HELP?")
        self.assertEqual(q, "HELLO THERE")
        self.assertEqual(r, "HEY! HOW CAN I HELP?")

    def test_strips_whitespace(self):
        q, r = gs.parse_pair_line("  HELLO  |  HI THERE  ")
        self.assertEqual(q, "HELLO")
        self.assertEqual(r, "HI THERE")

    def test_no_pipe_returns_none(self):
        self.assertIsNone(gs.parse_pair_line("HELLO THERE NO PIPE"))

    def test_empty_side_returns_none(self):
        self.assertIsNone(gs.parse_pair_line("|RESPONSE ONLY"))
        self.assertIsNone(gs.parse_pair_line("QUERY ONLY|"))

    def test_strips_quotes(self):
        q, r = gs.parse_pair_line('"HOW ARE YOU?"|"DOING GREAT!"')
        self.assertEqual(q, "HOW ARE YOU?")
        self.assertEqual(r, "DOING GREAT!")

    def test_uppercase_conversion(self):
        q, r = gs.parse_pair_line("hello there|hey how are you")
        self.assertEqual(q, "HELLO THERE")
        self.assertEqual(r, "HEY HOW ARE YOU")


class ValidatePair(unittest.TestCase):

    def test_valid_pair_passes(self):
        self.assertTrue(gs.is_valid_pair("HOW ARE YOU?", "DOING GREAT, THANKS!"))

    def test_too_short_query_fails(self):
        self.assertFalse(gs.is_valid_pair("HI", "DOING GREAT!"))

    def test_too_long_query_fails(self):
        self.assertFalse(gs.is_valid_pair("A" * 91, "RESPONSE"))

    def test_too_long_response_fails(self):
        self.assertFalse(gs.is_valid_pair("HOW ARE YOU?", "A" * 121))

    def test_non_ascii_fails(self):
        self.assertFalse(gs.is_valid_pair("HOW ARE YOU?", "HÉLLO THERE"))

    def test_pipe_in_content_fails(self):
        self.assertFalse(gs.is_valid_pair("HOW|ARE YOU?", "FINE"))
        self.assertFalse(gs.is_valid_pair("HOW ARE YOU?", "FINE|GOOD"))

    def test_ctx_overflow_fails(self):
        # Q + SEP + R + EOS must fit in 128
        long_q = "A" * 60
        long_r = "B" * 70   # 60+1+70+1 = 132 > 128
        self.assertFalse(gs.is_valid_pair(long_q, long_r, max_ctx=128))

    def test_ctx_fits_passes(self):
        q = "A" * 60
        r = "B" * 60   # 60+1+60+1 = 122 < 128
        self.assertTrue(gs.is_valid_pair(q, r, max_ctx=128))


class BuildPrompt(unittest.TestCase):

    def test_returns_string(self):
        prompt = gs.build_scenario_prompt("a user greets an AI in the morning")
        self.assertIsInstance(prompt, str)
        self.assertIn("QUERY|RESPONSE", prompt)

    def test_includes_scenario(self):
        prompt = gs.build_scenario_prompt("a user asks an AI about its feelings")
        self.assertIn("feelings", prompt)


if __name__ == "__main__":
    unittest.main()
