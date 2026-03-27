import unittest
from pathlib import Path
import tempfile
import json

class TestSpeculativeRegistry(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.papers_path = Path(self.tempdir.name) / "papers.json"
        self.profiles_path = Path(self.tempdir.name) / "profiles.json"
        self.prompt_sources_path = Path(self.tempdir.name) / "prompt_sources.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def write_json(self, path, obj):
        path.write_text(json.dumps(obj))

    def test_rejects_approximate_paper_without_gap(self):
        papers = {"p1": {"support_level": "approximate", "paper_id": "p1"}}
        self.write_json(self.papers_path, papers)
        profiles = {"profile1": {"profile_tier": "promotion", "batching_mode": "static", "streaming_mode": "streaming", "stop_condition": "max_tokens", "request_order": "fixed", "profile_id": "profile1"}}
        self.write_json(self.profiles_path, profiles)
        self.write_json(self.prompt_sources_path, {"p1": ["prompt1"]})
        from speculative_registry import load_registry_bundle
        with self.assertRaises(ValueError) as ctx:
            load_registry_bundle(papers_path=self.papers_path, profiles_path=self.profiles_path, prompt_sources_path=self.prompt_sources_path)
        self.assertIn("approximation_gap", str(ctx.exception))

    def test_rejects_unknown_profile_enum(self):
        papers = {"p1": {"support_level": "native", "paper_id": "p1"}}
        self.write_json(self.papers_path, papers)
        profiles = {"profile1": {"profile_tier": "unknown", "batching_mode": "static", "streaming_mode": "streaming", "stop_condition": "max_tokens", "request_order": "fixed", "profile_id": "profile1"}}
        self.write_json(self.profiles_path, profiles)
        self.write_json(self.prompt_sources_path, {"p1": ["prompt1"]})
        from speculative_registry import load_registry_bundle
        with self.assertRaises(ValueError) as ctx:
            load_registry_bundle(papers_path=self.papers_path, profiles_path=self.profiles_path, prompt_sources_path=self.prompt_sources_path)
        self.assertIn("profile_tier", str(ctx.exception))

    def test_accepts_native_paper_without_gap(self):
        papers = {"p1": {"support_level": "native", "paper_id": "p1"}}
        self.write_json(self.papers_path, papers)
        profiles = {"profile1": {"profile_tier": "promotion", "batching_mode": "static", "streaming_mode": "streaming", "stop_condition": "max_tokens", "request_order": "fixed", "profile_id": "profile1"}}
        self.write_json(self.profiles_path, profiles)
        self.write_json(self.prompt_sources_path, {"p1": ["prompt1"]})
        from speculative_registry import load_registry_bundle
        bundle = load_registry_bundle(papers_path=self.papers_path, profiles_path=self.profiles_path, prompt_sources_path=self.prompt_sources_path)
        self.assertIn("p1", bundle.papers)
        self.assertIn("profile1", bundle.profiles)
        self.assertEqual(bundle.prompt_sources["p1"], ["prompt1"])

    def test_rejects_unsupported_paper_without_blocker(self):
        papers = {"p1": {"support_level": "unsupported", "paper_id": "p1"}}
        self.write_json(self.papers_path, papers)
        profiles = {"profile1": {"profile_tier": "promotion", "batching_mode": "static", "streaming_mode": "streaming", "stop_condition": "max_tokens", "request_order": "fixed", "profile_id": "profile1"}}
        self.write_json(self.profiles_path, profiles)
        self.write_json(self.prompt_sources_path, {"p1": ["prompt1"]})
        from speculative_registry import load_registry_bundle
        with self.assertRaises(ValueError) as ctx:
            load_registry_bundle(papers_path=self.papers_path, profiles_path=self.profiles_path, prompt_sources_path=self.prompt_sources_path)
        self.assertIn("blocker_reason", str(ctx.exception))

    def test_rejects_unknown_request_order(self):
        papers = {"p1": {"support_level": "native", "paper_id": "p1"}}
        self.write_json(self.papers_path, papers)
        profiles = {"profile1": {"profile_tier": "promotion", "batching_mode": "static", "streaming_mode": "streaming", "stop_condition": "max_tokens", "request_order": "random", "profile_id": "profile1"}}
        self.write_json(self.profiles_path, profiles)
        self.write_json(self.prompt_sources_path, {"p1": ["prompt1"]})
        from speculative_registry import load_registry_bundle
        with self.assertRaises(ValueError) as ctx:
            load_registry_bundle(papers_path=self.papers_path, profiles_path=self.profiles_path, prompt_sources_path=self.prompt_sources_path)
        self.assertIn("request_order", str(ctx.exception))
