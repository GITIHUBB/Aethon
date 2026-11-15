import unittest
from unittest.mock import patch

from main import (
    MAX_DISTANCE,
    NASA_REFERENCES_HEADING,
    QueryRequest,
    _ACRONYM_EXPANSIONS,
    _clean_ai_response,
    _small_talk_route,
    _strip_model_references,
    app,
    expand_and_normalize_query,
    grounding_match,
    query_agent,
    source_url,
)


class BackendFeatureTests(unittest.TestCase):
    def test_grounding_match_is_bounded_and_monotonic(self):
        scores = [grounding_match(distance) for distance in (0, 0.35, 0.7, 1.05, MAX_DISTANCE, 2)]

        self.assertEqual(scores[0], 98)
        self.assertEqual(scores[-2:], [70, 70])
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_small_talk_router_matches_standalone_phrases_only(self):
        self.assertIsNotNone(_small_talk_route("helo!"))
        self.assertIsNotNone(_small_talk_route("thanks"))
        self.assertIsNone(_small_talk_route("help me understand flame spread"))

    def test_model_sources_are_removed_for_server_owned_references(self):
        answer = (
            "The passage discusses flame spread [19930010989.pdf p.202].\n\n"
            f"{NASA_REFERENCES_HEADING}\n- fabricated source"
        )
        bold_heading_answer = (
            "Grounded summary.\n\n**References**\n- fabricated source"
        )

        self.assertEqual(
            _strip_model_references(answer),
            "The passage discusses flame spread.",
        )
        self.assertEqual(
            _strip_model_references(bold_heading_answer),
            "Grounded summary.",
        )

    def test_short_queries_and_supported_acronyms_expand_for_search(self):
        for acronym, phrase in _ACRONYM_EXPANSIONS.items():
            with self.subTest(acronym=acronym):
                self.assertIn(phrase, expand_and_normalize_query(acronym))

        self.assertIn("NASA microgravity combustion", expand_and_normalize_query("extinction"))
        self.assertNotIn("Smoke Aerosol Measurement", expand_and_normalize_query("the same material"))

    def test_response_cleanup_removes_role_tags_and_robotic_prefixes(self):
        answer = (
            "<|assistant|>As an AI language model, here's the answer.\n"
            "The passage discusses flame spread [19930010989.pdf p.202].\n"
            "### References\n- fabricated source"
        )

        cleaned = _clean_ai_response(answer)

        self.assertNotIn("<|assistant|>", cleaned)
        self.assertNotIn("As an AI language model", cleaned)
        self.assertNotIn("19930010989.pdf", cleaned)
        self.assertNotIn("fabricated source", cleaned)
        self.assertIn("The passage discusses flame spread", cleaned)

    def test_citation_urls_prefer_doi_and_use_direct_ntrs_ids(self):
        self.assertEqual(
            source_url("paper.pdf", {"doi": "10.1234/combustion.test"}),
            "https://doi.org/10.1234/combustion.test",
        )
        self.assertEqual(
            source_url("19930010989.pdf"),
            "https://ntrs.nasa.gov/citations/19930010989",
        )

    def test_query_response_owns_a_clickable_final_reference_section(self):
        docs = [
            {
                "metadata": {
                    "source": "19930010989.pdf",
                    "page": 7,
                    "doi": "10.1234/combustion.test",
                },
                "page_content": "PMMA flame spread in microgravity.",
                "distance": 0.2,
            }
        ]
        generated = (
            "**Findings**\n\n- Grounded answer [19930010989.pdf p.7]\n"
            "### References\n- fabricated source"
        )

        with (
            patch("main.similarity_search", return_value=docs) as search,
            patch("main._primary_key", return_value="test-key"),
            patch("main._backup_key", return_value=None),
            patch("main._ai_answer", return_value=(generated, True, "groq")),
        ):
            result = query_agent(QueryRequest(question="PMMA"))

        self.assertIn("polymethyl methacrylate", search.call_args.args[0])
        self.assertEqual(result["answer"].count(NASA_REFERENCES_HEADING), 1)
        self.assertIn(
            "- [19930010989.pdf (p. 7)](https://doi.org/10.1234/combustion.test)",
            result["answer"],
        )
        self.assertNotIn("19930010989.pdf p.7", result["answer"])
        self.assertNotIn("fabricated source", result["answer"])

    def test_chat_history_defaults_are_independent_lists(self):
        first = QueryRequest()
        second = QueryRequest()
        first.chat_history.append({"role": "user", "content": "hello"})

        self.assertEqual(second.chat_history, [])

    def test_health_route_is_registered(self):
        self.assertIn("/health", {route.path for route in app.routes})


if __name__ == "__main__":
    unittest.main()
