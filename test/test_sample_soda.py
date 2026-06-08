#!/usr/bin/env python3
"""Tests for py/sample_soda.py — stratified Wisp training set sampler."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
import sample_soda as ss


class ClassifyPair(unittest.TestCase):

    def test_greeting_hello(self):
        self.assertEqual(ss.classify("HELLO", "HEY THERE!"), "greetings")

    def test_greeting_hi(self):
        self.assertEqual(ss.classify("HI THERE", "HEY!"), "greetings")

    def test_farewell(self):
        self.assertEqual(ss.classify("GOODBYE", "TAKE CARE!"), "greetings")

    def test_goodbye_see_you(self):
        self.assertEqual(ss.classify("SEE YOU LATER", "BYE!"), "greetings")

    def test_emotional_sad(self):
        self.assertEqual(ss.classify("I'M REALLY SAD TODAY", "SORRY TO HEAR THAT."), "emotional")

    def test_emotional_excited(self):
        self.assertEqual(ss.classify("I'M SO EXCITED!", "THAT'S AWESOME!"), "emotional")

    def test_emotional_reaction_congrats(self):
        self.assertEqual(ss.classify("I JUST GOT PROMOTED!", "CONGRATS!"), "emotional")

    def test_joke_request(self):
        self.assertEqual(ss.classify("TELL ME A JOKE", "WHY DID THE CHICKEN..."), "jokes")

    def test_joke_knock_knock(self):
        self.assertEqual(ss.classify("KNOCK KNOCK", "WHO'S THERE?"), "jokes")

    def test_joke_why_did(self):
        self.assertEqual(ss.classify("WHY DID THE SCARECROW WIN?", "OUTSTANDING IN HIS FIELD!"), "jokes")

    def test_opinion_prefer(self):
        self.assertEqual(ss.classify("DO YOU PREFER CATS OR DOGS?", "DOGS FOR SURE!"), "opinions")

    def test_opinion_favorite(self):
        self.assertEqual(ss.classify("WHAT'S YOUR FAVORITE FOOD?", "PIZZA!"), "opinions")

    def test_opinion_or(self):
        self.assertEqual(ss.classify("COFFEE OR TEA?", "COFFEE EVERY TIME!"), "opinions")

    def test_meta_who(self):
        self.assertEqual(ss.classify("WHO ARE YOU?", "JUST A FRIENDLY AI!"), "meta")

    def test_meta_what_are_you(self):
        self.assertEqual(ss.classify("WHAT ARE YOU?", "I'M AN AI ASSISTANT."), "meta")

    def test_small_talk_whats_up(self):
        # "WHAT'S UP" matches greetings (higher priority) — that's correct
        self.assertEqual(ss.classify("WHAT'S UP?", "NOT MUCH, YOU?"), "greetings")

    def test_small_talk_long_day(self):
        self.assertEqual(ss.classify("LONG DAY?", "YEAH, EXHAUSTING."), "small_talk")

    def test_reaction_agreement(self):
        self.assertEqual(ss.classify("I KNOW RIGHT?", "TOTALLY!"), "reactions")

    def test_reaction_isnt_it(self):
        self.assertEqual(ss.classify("ISN'T IT AMAZING?", "IT REALLY IS!"), "reactions")

    def test_unclassified_returns_none(self):
        self.assertIsNone(ss.classify("YADIER STOP DRINKING NOW", "LEAVE ME ALONE"))

    def test_character_specific_rejected(self):
        # Queries with proper names addressing a specific person → None
        self.assertIsNone(ss.classify("DIANE WHAT ARE YOU DOING HERE", "I LIVE HERE"))

    def test_greetings_take_priority_over_emotion(self):
        # "HOW ARE YOU" is greeting, not emotional
        self.assertEqual(ss.classify("HOW ARE YOU?", "DOING GREAT!"), "greetings")


class SampleStrata(unittest.TestCase):

    def _make_pairs(self, labels_and_pairs):
        """Return [(q, r, stratum)] test data."""
        return [(q, r, s) for s, q, r in labels_and_pairs]

    def test_returns_at_most_target_per_stratum(self):
        pairs = [("HELLO", "HI!"), ("HI", "HEY!"), ("HEY", "HELLO!"),
                 ("TELL ME A JOKE", "WHY DID..."), ("KNOCK KNOCK", "WHO?")]
        result = ss.sample_strata(pairs, targets={"greetings": 2, "jokes": 10})
        counts = {}
        for _, _, s in result:
            counts[s] = counts.get(s, 0) + 1
        self.assertLessEqual(counts.get("greetings", 0), 2)

    def test_includes_all_strata_present(self):
        pairs = [("HELLO", "HI!"), ("TELL ME A JOKE", "WHY DID..."),
                 ("I'M SAD", "SORRY!"), ("WHO ARE YOU", "AN AI!")]
        result = ss.sample_strata(pairs)
        strata_found = {s for _, _, s in result}
        self.assertIn("greetings", strata_found)
        self.assertIn("jokes", strata_found)
        self.assertIn("emotional", strata_found)
        self.assertIn("meta", strata_found)

    def test_output_is_shuffled_not_stratum_grouped(self):
        # With enough pairs the output shouldn't be all one stratum first
        import random; random.seed(0)
        pairs = [("HELLO", "HI!"), ("HI", "HEY!"), ("TELL ME A JOKE", "WHY DID..."),
                 ("KNOCK KNOCK", "WHO?"), ("I'M SAD", "SORRY!"), ("WHO ARE YOU", "AN AI!")]
        result = ss.sample_strata(pairs, seed=42)
        strata_sequence = [s for _, _, s in result]
        # Should not be sorted/grouped
        self.assertFalse(strata_sequence == sorted(strata_sequence),
                         "Output should not be stratum-sorted")


if __name__ == "__main__":
    unittest.main()
