#!/usr/bin/env python3
"""
Tests for expand_data.py — pure (non-network) functions only.

Covers:
- Template parsing and built-in fallback
- Slot filling (known slots, unknown slots left as-is, deduplication)
- Length-bucket sampling distribution
- Response filtering (len(q)+len(r)+1 <= max_ctx)
- Checkpoint save/load round-trip
- Defensive query line parsing

Run: python3 test/test_expand_data.py  (or pytest test/test_expand_data.py)
"""

import json
import os
import random
import sys
import tempfile
import unittest

# Ensure the py/ directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))

from expand_data import (
    BUILTIN_TEMPLATES,
    SLOT_WORDS,
    LENGTH_BUCKETS,
    clean_line,
    fill_slots,
    is_preamble,
    load_checkpoint,
    parse_lines,
    phase1_templates,
    phase3_slot_fill,
    sample_length_instruction,
    save_checkpoint,
)


# ─── Template parsing ─────────────────────────────────────────────────────────

class TestTemplateParsing(unittest.TestCase):

    def test_builtin_fallback_when_no_file(self):
        """phase1_templates returns BUILTIN_TEMPLATES when no file given."""
        templates = phase1_templates(None)
        self.assertEqual(templates, list(BUILTIN_TEMPLATES))
        self.assertGreater(len(templates), 0)

    def test_builtin_fallback_when_file_missing(self):
        """phase1_templates falls back to built-ins when file does not exist."""
        templates = phase1_templates("/nonexistent/path/templates.txt")
        self.assertEqual(templates, list(BUILTIN_TEMPLATES))

    def test_load_from_file(self):
        """phase1_templates loads templates from a given file, one per line."""
        lines = [
            "Tell me about {topic}",
            "What is the capital of {country}",
            "How do I {verb} a {noun}",
            "",  # blank lines should be ignored
            "  Recommend a {genre}  ",  # whitespace should be stripped
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            tmp_path = f.name

        try:
            templates = phase1_templates(tmp_path)
        finally:
            os.unlink(tmp_path)

        # Blank line excluded; whitespace-stripped
        self.assertIn("Tell me about {topic}", templates)
        self.assertIn("What is the capital of {country}", templates)
        self.assertIn("Recommend a {genre}", templates)
        self.assertNotIn("", templates)
        self.assertEqual(len(templates), 4)  # 4 non-blank lines

    def test_builtin_has_no_code_or_math(self):
        """Built-in templates should not contain code/math categories."""
        forbidden_keywords = ["code", "program", "function", "algorithm",
                              "math", "equation", "formula", "calculate"]
        for template in BUILTIN_TEMPLATES:
            low = template.lower()
            for kw in forbidden_keywords:
                self.assertNotIn(kw, low,
                    f"Built-in template contains forbidden keyword '{kw}': {template}")

    def test_builtin_templates_have_required_categories(self):
        """Built-in templates should span multiple categories."""
        all_text = " ".join(BUILTIN_TEMPLATES).lower()
        # Check that a variety of placeholders exist
        self.assertIn("{topic}", all_text)
        self.assertIn("{country}", all_text)
        self.assertIn("{verb}", all_text)
        self.assertIn("{noun}", all_text)

    def test_builtin_count_at_least_40(self):
        """Built-in set must have at least 40 templates."""
        self.assertGreaterEqual(len(BUILTIN_TEMPLATES), 40)


# ─── Line cleaning / parsing ──────────────────────────────────────────────────

class TestLineParsing(unittest.TestCase):

    def test_strip_numbering_dot(self):
        self.assertEqual(clean_line("1. Hello there"), "Hello there")

    def test_strip_numbering_paren(self):
        self.assertEqual(clean_line("3) Tell me a joke"), "Tell me a joke")

    def test_strip_bullet_dash(self):
        self.assertEqual(clean_line("- What is your name"), "What is your name")

    def test_strip_bullet_star(self):
        self.assertEqual(clean_line("* Give me advice"), "Give me advice")

    def test_strip_bullet_dot(self):
        self.assertEqual(clean_line("• Recommend a movie"), "Recommend a movie")

    def test_strip_quotes(self):
        self.assertEqual(clean_line('"Tell me a joke"'), "Tell me a joke")

    def test_strip_single_quotes(self):
        self.assertEqual(clean_line("'What is your name'"), "What is your name")

    def test_strip_whitespace(self):
        self.assertEqual(clean_line("  hello world  "), "hello world")

    def test_is_preamble_here_are(self):
        self.assertTrue(is_preamble("Here are 20 templates:"))

    def test_is_preamble_sure(self):
        self.assertTrue(is_preamble("Sure, here you go!"))

    def test_is_preamble_of_course(self):
        self.assertTrue(is_preamble("Of course! I'd be happy to help."))

    def test_is_preamble_certainly(self):
        self.assertTrue(is_preamble("Certainly, here are some options:"))

    def test_not_preamble_real_template(self):
        self.assertFalse(is_preamble("Tell me a joke about {topic}"))

    def test_parse_lines_strips_numbering(self):
        text = "1. Tell me a joke\n2. What is your name\n3. How are you"
        lines = parse_lines(text)
        self.assertIn("Tell me a joke", lines)
        self.assertIn("What is your name", lines)
        self.assertIn("How are you", lines)

    def test_parse_lines_skips_preambles(self):
        text = "Here are some templates:\n1. Tell me a joke\n2. What is your name"
        lines = parse_lines(text)
        # Preamble line should be excluded
        self.assertNotIn("Here are some templates:", lines)
        self.assertIn("Tell me a joke", lines)

    def test_parse_lines_skips_empty(self):
        text = "Tell me a joke\n\n\nWhat is your name"
        lines = parse_lines(text)
        self.assertNotIn("", lines)
        self.assertEqual(len(lines), 2)

    def test_parse_lines_skips_non_alpha(self):
        text = "Tell me a joke\n---\n12345\nWhat is your name"
        lines = parse_lines(text)
        self.assertNotIn("---", lines)
        self.assertNotIn("12345", lines)

    def test_parse_lines_skips_sure_preambles(self):
        text = "Sure, happy to help!\nTell me a joke\nAbsolutely, here we go!"
        lines = parse_lines(text)
        self.assertEqual(lines, ["Tell me a joke"])


# ─── Slot filling ─────────────────────────────────────────────────────────────

class TestSlotFilling(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(123)

    def test_fill_topic_slot(self):
        result = fill_slots("Tell me about {topic}", self.rng)
        self.assertNotIn("{topic}", result)
        # Should be a known topic word
        found = any(topic in result for topic in SLOT_WORDS["topic"])
        self.assertTrue(found, f"No topic word found in: {result}")

    def test_fill_verb_slot(self):
        result = fill_slots("How do I {verb} a thing", self.rng)
        self.assertNotIn("{verb}", result)
        found = any(verb in result for verb in SLOT_WORDS["verb"])
        self.assertTrue(found, f"No verb found in: {result}")

    def test_fill_noun_slot(self):
        result = fill_slots("How do I fix a {noun}", self.rng)
        self.assertNotIn("{noun}", result)
        found = any(noun in result for noun in SLOT_WORDS["noun"])
        self.assertTrue(found, f"No noun found in: {result}")

    def test_fill_country_slot(self):
        result = fill_slots("What is the capital of {country}", self.rng)
        self.assertNotIn("{country}", result)
        found = any(c in result for c in SLOT_WORDS["country"])
        self.assertTrue(found, f"No country found in: {result}")

    def test_fill_genre_slot(self):
        result = fill_slots("Recommend a {genre} for tonight", self.rng)
        self.assertNotIn("{genre}", result)

    def test_fill_occasion_slot(self):
        result = fill_slots("Suggest something for a {occasion}", self.rng)
        self.assertNotIn("{occasion}", result)

    def test_unknown_slot_left_as_is(self):
        """Unknown {placeholder} should be left intact."""
        result = fill_slots("Tell me about {foobar}", self.rng)
        self.assertIn("{foobar}", result)

    def test_multiple_slots_filled(self):
        result = fill_slots("How do I {verb} a {noun}", self.rng)
        self.assertNotIn("{verb}", result)
        self.assertNotIn("{noun}", result)

    def test_no_slots(self):
        template = "How are you today"
        result = fill_slots(template, self.rng)
        self.assertEqual(result, template)

    def test_deduplication_in_phase3(self):
        """phase3_slot_fill should deduplicate results."""
        # Use a small template set with one slot and limited words to force collisions
        templates = ["Tell me about {topic}"] * 3
        # Override SLOT_WORDS temporarily so only 2 topic words exist
        import expand_data
        original = expand_data.SLOT_WORDS["topic"]
        expand_data.SLOT_WORDS["topic"] = ["dogs", "cats"]
        try:
            prompts = phase3_slot_fill(templates, samples_per_template=5, rng=random.Random(0))
            # With only 2 options, we can't get more than 2 unique prompts per template
            # But since all 3 templates are the same, total unique is still 2
            unique = len(set(p.lower() for p in prompts))
            self.assertEqual(unique, len(prompts), "Duplicates found in phase3 output")
        finally:
            expand_data.SLOT_WORDS["topic"] = original

    def test_no_slot_template_deduped_across_multiple_uses(self):
        """Templates with no slots should appear at most once."""
        templates = ["How are you"] * 5
        prompts = phase3_slot_fill(templates, samples_per_template=5, rng=random.Random(0))
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0], "How are you")


# ─── Length bucket sampling ───────────────────────────────────────────────────

class TestLengthBuckets(unittest.TestCase):

    def test_sample_returns_known_instruction(self):
        """sample_length_instruction must return one of the known instructions."""
        known = {instr for _, instr in LENGTH_BUCKETS}
        rng = random.Random(0)
        for _ in range(100):
            result = sample_length_instruction(rng)
            self.assertIn(result, known)

    def test_distribution_approx(self):
        """Over many draws, verify ~40/35/20/5% distribution."""
        rng = random.Random(42)
        n = 10000
        counts: dict[str, int] = {}
        for _ in range(n):
            instr = sample_length_instruction(rng)
            counts[instr] = counts.get(instr, 0) + 1

        expected = {instr: prob for prob, instr in LENGTH_BUCKETS}
        for instr, prob in expected.items():
            observed = counts.get(instr, 0) / n
            # Allow ±3% tolerance
            self.assertAlmostEqual(
                observed, prob, delta=0.03,
                msg=f"Bucket '{instr[:30]}': expected ~{prob:.0%}, got {observed:.1%}"
            )

    def test_bucket_probabilities_sum_to_one(self):
        """LENGTH_BUCKETS probabilities must sum to 1.0."""
        total = sum(p for p, _ in LENGTH_BUCKETS)
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_terse_bucket_is_most_common(self):
        """Terse (40%) should be the most frequently sampled bucket."""
        rng = random.Random(7)
        n = 5000
        counts: dict[str, int] = {}
        for _ in range(n):
            instr = sample_length_instruction(rng)
            counts[instr] = counts.get(instr, 0) + 1

        # The terse bucket instruction
        terse_instr = LENGTH_BUCKETS[0][1]  # first entry
        max_bucket = max(counts, key=lambda k: counts[k])
        self.assertEqual(max_bucket, terse_instr)


# ─── Response filtering ───────────────────────────────────────────────────────

class TestResponseFiltering(unittest.TestCase):
    """
    Tests for the length filter: len(q) + len(r) + 1 <= max_ctx.
    We verify the filter logic directly without making network calls.
    """

    def _apply_filter(self, q: str, r: str, max_ctx: int) -> bool:
        """Mirror the filter in _generate_response."""
        return len(q) + len(r) + 1 <= max_ctx

    def test_fits_within_ctx(self):
        q = "HELLO HOW ARE YOU"
        r = "I AM FINE THANKS"
        # len=17, len=16, +1 = 34 <= 256
        self.assertTrue(self._apply_filter(q, r, 256))

    def test_exactly_at_ctx_limit(self):
        q = "A" * 100
        r = "B" * 155  # 100 + 155 + 1 = 256
        self.assertTrue(self._apply_filter(q, r, 256))

    def test_one_over_limit(self):
        q = "A" * 100
        r = "B" * 156  # 100 + 156 + 1 = 257 > 256
        self.assertFalse(self._apply_filter(q, r, 256))

    def test_wisp_ctx_64(self):
        q = "HELLO"   # 5
        r = "HI THERE"  # 8; 5+8+1=14 <= 64
        self.assertTrue(self._apply_filter(q, r, 64))

    def test_wisp_ctx_64_overflow(self):
        q = "A" * 30
        r = "B" * 35  # 30+35+1=66 > 64
        self.assertFalse(self._apply_filter(q, r, 64))

    def test_shade_ctx_128(self):
        q = "A" * 60
        r = "B" * 67  # 60+67+1=128 — exactly at limit
        self.assertTrue(self._apply_filter(q, r, 128))

    def test_shade_ctx_128_overflow(self):
        q = "A" * 60
        r = "B" * 68  # 60+68+1=129 > 128
        self.assertFalse(self._apply_filter(q, r, 128))

    def test_empty_strings(self):
        self.assertTrue(self._apply_filter("", "", 256))  # 0+0+1=1 <= 256
        self.assertFalse(self._apply_filter("", "", 0))   # 0+0+1=1 > 0


# ─── Checkpoint save/load ─────────────────────────────────────────────────────

class TestCheckpoint(unittest.TestCase):

    def test_save_and_load_round_trip(self):
        state = {
            "completed_phases": [1, 2, 3],
            "data": {
                "templates": ["Tell me about {topic}", "How do I {verb} a {noun}"],
                "seed_prompts": ["Tell me about dogs", "How do I cook a cake"],
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, state)
            loaded = load_checkpoint(tmp_path)
            self.assertEqual(loaded, state)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_returns_empty(self):
        state = load_checkpoint("/nonexistent/path/checkpoint.json")
        self.assertEqual(state, {"completed_phases": [], "data": {}})

    def test_save_is_atomic(self):
        """save_checkpoint writes to .tmp then renames — no partial file."""
        state = {"completed_phases": [1], "data": {"templates": ["a", "b", "c"]}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, state)
            # The .tmp file should NOT exist after save (it was renamed)
            self.assertFalse(os.path.exists(tmp_path + ".tmp"))
            # The actual file should exist
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            os.unlink(tmp_path)

    def test_checkpoint_preserves_pairs(self):
        """Pairs stored as lists of [q, r] (JSON) round-trip correctly."""
        pairs = [["HELLO", "HI THERE"], ["HOW ARE YOU", "IM FINE"]]
        state = {
            "completed_phases": [1, 2, 3, 4, 5],
            "data": {"phase5_pairs": pairs},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, state)
            loaded = load_checkpoint(tmp_path)
            self.assertEqual(loaded["data"]["phase5_pairs"], pairs)
        finally:
            os.unlink(tmp_path)

    def test_completed_phases_tracks_progress(self):
        """After saving phases 1 and 2, both should appear in completed_phases."""
        state = {"completed_phases": [], "data": {}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            # Simulate phase 1 completion
            state["completed_phases"] = [1]
            save_checkpoint(tmp_path, state)

            # Simulate phase 2 completion
            state["completed_phases"] = [1, 2]
            save_checkpoint(tmp_path, state)

            loaded = load_checkpoint(tmp_path)
            self.assertIn(1, loaded["completed_phases"])
            self.assertIn(2, loaded["completed_phases"])
        finally:
            os.unlink(tmp_path)


# ─── Word list completeness ───────────────────────────────────────────────────

class TestWordLists(unittest.TestCase):

    def test_topic_has_at_least_80_words(self):
        self.assertGreaterEqual(len(SLOT_WORDS["topic"]), 80)

    def test_verb_has_at_least_60_words(self):
        self.assertGreaterEqual(len(SLOT_WORDS["verb"]), 60)

    def test_noun_has_at_least_60_words(self):
        self.assertGreaterEqual(len(SLOT_WORDS["noun"]), 60)

    def test_country_has_at_least_50_entries(self):
        self.assertGreaterEqual(len(SLOT_WORDS["country"]), 50)

    def test_genre_has_at_least_30_entries(self):
        self.assertGreaterEqual(len(SLOT_WORDS["genre"]), 30)

    def test_occasion_has_at_least_20_entries(self):
        self.assertGreaterEqual(len(SLOT_WORDS["occasion"]), 20)

    def test_no_duplicates_in_word_lists(self):
        for slot, words in SLOT_WORDS.items():
            self.assertEqual(
                len(words), len(set(words)),
                f"Duplicates found in SLOT_WORDS['{slot}']"
            )

    def test_all_word_list_entries_non_empty(self):
        for slot, words in SLOT_WORDS.items():
            for w in words:
                self.assertTrue(w.strip(), f"Empty word in SLOT_WORDS['{slot}']")


# ─── Integration: phase3 output properties ───────────────────────────────────

class TestPhase3Output(unittest.TestCase):

    def test_output_is_list_of_strings(self):
        templates = BUILTIN_TEMPLATES[:5]
        prompts = phase3_slot_fill(templates, samples_per_template=2, rng=random.Random(0))
        for p in prompts:
            self.assertIsInstance(p, str)

    def test_no_unfilled_known_slots_in_output(self):
        """All known {slot} placeholders should be replaced."""
        templates = [
            "Tell me about {topic}",
            "What is the capital of {country}",
            "How do I {verb} a {noun}",
            "Recommend a {genre} for {occasion}",
        ]
        prompts = phase3_slot_fill(templates, samples_per_template=3, rng=random.Random(0))
        for p in prompts:
            for slot in ["topic", "country", "verb", "noun", "genre", "occasion"]:
                self.assertNotIn(
                    "{" + slot + "}", p,
                    f"Unfilled slot {{{slot}}} found in: {p}"
                )

    def test_samples_per_template_respected(self):
        """Each template should produce at most samples_per_template unique prompts."""
        # Use a template with many possible fills so we don't hit dedup limit
        templates = ["Tell me about {topic}"]
        n = 5
        prompts = phase3_slot_fill(templates, samples_per_template=n, rng=random.Random(0))
        self.assertLessEqual(len(prompts), n)

    def test_all_prompts_are_unique(self):
        templates = BUILTIN_TEMPLATES[:10]
        prompts = phase3_slot_fill(templates, samples_per_template=3, rng=random.Random(0))
        self.assertEqual(len(prompts), len(set(p.lower() for p in prompts)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
