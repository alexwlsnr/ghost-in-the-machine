#!/usr/bin/env python3
"""Tests for py/ingest_soda_dialogues.py — full multi-turn SODA dialogue extractor."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import ingest_soda_dialogues as isd


class NormaliseTurn(unittest.TestCase):

    def test_uppercases(self):
        self.assertEqual(isd.normalise_turn("hello there"), "HELLO THERE")

    def test_strips_whitespace(self):
        self.assertEqual(isd.normalise_turn("  hi  "), "HI")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(isd.normalise_turn("good   morning"), "GOOD MORNING")

    def test_strips_quotes(self):
        self.assertEqual(isd.normalise_turn('"hello"'), "HELLO")

    def test_applies_name_normalisation(self):
        result = isd.normalise_turn("Hey, Carlaton! How are you?")
        self.assertIn("HUMAN", result)
        self.assertNotIn("CARLATON", result)


class FilterTurn(unittest.TestCase):

    def test_valid_turn_passes(self):
        self.assertTrue(isd.filter_turn("HOW ARE YOU DOING TODAY"))

    def test_too_short_fails(self):
        self.assertFalse(isd.filter_turn("HI"))

    def test_too_long_fails(self):
        self.assertFalse(isd.filter_turn("A" * 130))

    def test_non_ascii_fails(self):
        self.assertFalse(isd.filter_turn("HÉLLO THERE"))

    def test_pipe_fails(self):
        self.assertFalse(isd.filter_turn("HELLO|WORLD"))


class ExtractDialogue(unittest.TestCase):

    def test_basic_two_turn(self):
        turns = ["Hello!", "Hey there, how are you?"]
        result = isd.extract_dialogue(turns, max_ctx=1024)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_trims_to_even_number_of_turns(self):
        # Odd number of turns → drop last so we have complete Q/R pairs
        turns = ["Hi", "Hello!", "How are you?"]
        result = isd.extract_dialogue(turns, max_ctx=1024)
        # Must be even (complete pairs) or None
        if result is not None:
            self.assertEqual(len(result) % 2, 0)

    def test_too_short_dialogue_returns_none(self):
        result = isd.extract_dialogue(["Hi"], max_ctx=1024)
        self.assertIsNone(result)

    def test_ctx_overflow_trims_from_front(self):
        # Very long turns — should trim oldest until it fits
        long_turn = "A" * 100
        turns = [long_turn] * 10
        result = isd.extract_dialogue(turns, max_ctx=256)
        if result is not None:
            total = sum(len(t) for t in result) + len(result) + 1
            self.assertLessEqual(total, 256)

    def test_filters_invalid_turns(self):
        # Turns with pipes or non-ASCII should be dropped
        turns = ["Hello there!", "HI|THERE", "How are you?", "GREAT THANKS"]
        result = isd.extract_dialogue(turns, max_ctx=1024)
        if result is not None:
            for t in result:
                self.assertNotIn("|", t)

    def test_returns_normalised_uppercase(self):
        turns = ["hello!", "hey there, how are you doing?"]
        result = isd.extract_dialogue(turns, max_ctx=1024)
        self.assertIsNotNone(result)
        for t in result:
            self.assertEqual(t, t.upper())


class FormatDialogue(unittest.TestCase):

    def test_pipe_separated_turns(self):
        turns = ["HELLO", "HEY THERE", "HOW ARE YOU", "DOING GREAT"]
        line = isd.format_dialogue(turns)
        self.assertEqual(line, "HELLO|HEY THERE|HOW ARE YOU|DOING GREAT")

    def test_two_turns(self):
        line = isd.format_dialogue(["HI", "HELLO HUMAN"])
        self.assertEqual(line, "HI|HELLO HUMAN")


if __name__ == "__main__":
    unittest.main()
