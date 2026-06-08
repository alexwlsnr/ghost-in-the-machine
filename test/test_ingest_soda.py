#!/usr/bin/env python3
"""Tests for py/ingest_soda.py — pure functions only, no network calls."""

import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import ingest_soda as si


class FilterPair(unittest.TestCase):

    def test_accepts_clean_pair(self):
        self.assertTrue(si.filter_pair("HELLO", "HEY THERE!"))

    def test_rejects_non_ascii(self):
        self.assertFalse(si.filter_pair("HELLO", "HEY ’ THERE"))

    def test_rejects_query_too_short(self):
        self.assertFalse(si.filter_pair("HI", "HEY THERE!"))  # < MIN_Q_LEN

    def test_rejects_response_too_short(self):
        self.assertFalse(si.filter_pair("HELLO THERE", "OK"))  # < MIN_R_LEN

    def test_rejects_query_too_long(self):
        self.assertFalse(si.filter_pair("A" * 100, "SHORT RESPONSE HERE"))

    def test_rejects_response_too_long(self):
        self.assertFalse(si.filter_pair("HELLO", "A" * 200))

    def test_rejects_pair_exceeding_ctx(self):
        # len(q) + 1 + len(r) + 1 > max_ctx
        self.assertFalse(si.filter_pair("A" * 60, "B" * 70, max_ctx=128))

    def test_accepts_pair_within_ctx(self):
        self.assertTrue(si.filter_pair("A" * 30, "B" * 50, max_ctx=128))

    def test_rejects_empty_query(self):
        self.assertFalse(si.filter_pair("", "RESPONSE HERE"))

    def test_rejects_empty_response(self):
        self.assertFalse(si.filter_pair("HELLO THERE", ""))


class NormalisePair(unittest.TestCase):

    def test_uppercases_text(self):
        q, r = si.normalise_pair("hello friend", "hey there!")
        self.assertEqual(q, "HELLO FRIEND")
        self.assertEqual(r, "HEY THERE!")

    def test_strips_whitespace(self):
        q, r = si.normalise_pair("  hello  ", "  hi  ")
        self.assertEqual(q, "HELLO")
        self.assertEqual(r, "HI")

    def test_strips_surrounding_quotes(self):
        q, r = si.normalise_pair('"hello"', "'hey there'")
        self.assertEqual(q, "HELLO")
        self.assertEqual(r, "HEY THERE")

    def test_collapses_internal_whitespace(self):
        q, r = si.normalise_pair("hello   world", "hey  there")
        self.assertEqual(q, "HELLO WORLD")
        self.assertEqual(r, "HEY THERE")


class ExtractPairs(unittest.TestCase):

    def test_extracts_consecutive_turns(self):
        dialogue = ["Hi there", "Hey! How are you?", "I'm great", "Glad to hear it!"]
        pairs = si.extract_pairs(dialogue)
        # 3 consecutive pairs from 4 turns
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0], ("Hi there", "Hey! How are you?"))
        self.assertEqual(pairs[1], ("Hey! How are you?", "I'm great"))

    def test_empty_dialogue_gives_no_pairs(self):
        self.assertEqual(si.extract_pairs([]), [])

    def test_single_turn_gives_no_pairs(self):
        self.assertEqual(si.extract_pairs(["Hello"]), [])

    def test_two_turns_gives_one_pair(self):
        pairs = si.extract_pairs(["Hi", "Hello back"])
        self.assertEqual(len(pairs), 1)

    def test_skips_turns_with_pipe_char(self):
        # Pipe is our separator — turns containing | would corrupt the format
        dialogue = ["Hello|world", "Hi there", "Nice day", "Indeed!"]
        pairs = si.extract_pairs(dialogue)
        # First pair ("Hello|world", "Hi there") should be dropped
        valid = [(q, r) for q, r in pairs if "|" not in q and "|" not in r]
        self.assertEqual(len(valid), len(pairs))


class DeduplicatePairs(unittest.TestCase):

    def test_removes_exact_duplicates(self):
        pairs = [("HELLO", "HI"), ("HELLO", "HI"), ("BYE", "SEE YOU")]
        result = si.dedup_pairs(pairs)
        self.assertEqual(len(result), 2)

    def test_preserves_order(self):
        pairs = [("A", "B"), ("C", "D"), ("A", "B")]
        result = si.dedup_pairs(pairs)
        self.assertEqual(result[0], ("A", "B"))
        self.assertEqual(result[1], ("C", "D"))

    def test_deduplicates_by_query(self):
        # Same query, different responses -> keep first
        pairs = [("HELLO", "HI"), ("HELLO", "HEY THERE")]
        result = si.dedup_pairs(pairs, by_query=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "HI")  # first one kept


if __name__ == "__main__":
    unittest.main()
