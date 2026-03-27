from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json

SUPPORT_LEVELS = {"native", "approximate", "unsupported"}
PROFILE_TIERS = {"promotion", "smoke"}
BATCHING_MODES = {"static", "continuous"}
STREAMING_MODES = {"streaming", "non_streaming"}
STOP_CONDITIONS = {"max_tokens", "stop_strings"}
REQUEST_ORDERS = {"fixed", "seeded_shuffle"}

@dataclass
class PaperEntry:
    paper_id: str
    support_level: str
    approximation_gap: str = ""
    blocker_reason: str = ""

@dataclass
class ProfileEntry:
    profile_id: str
    profile_tier: str
    batching_mode: str
    streaming_mode: str
    stop_condition: str
    request_order: str

@dataclass
class RegistryBundle:
    papers: Dict[str, PaperEntry]
    profiles: Dict[str, ProfileEntry]
    prompt_sources: Dict[str, List[str]]


def load_registry_bundle(*, papers_path: Path, profiles_path: Path, prompt_sources_path: Path) -> RegistryBundle:
    papers_raw = json.loads(papers_path.read_text())
    profiles_raw = json.loads(profiles_path.read_text())
    prompt_sources = json.loads(prompt_sources_path.read_text())

    papers = {}
    for pid, pdata in papers_raw.items():
        support_level = pdata.get("support_level")
        if support_level not in SUPPORT_LEVELS:
            raise ValueError(f"Invalid support_level: {support_level}")
        if support_level == "approximate":
            gap = pdata.get("approximation_gap", "")
            if not gap:
                raise ValueError("approximation_gap required for approximate paper")
        if support_level == "unsupported":
            blocker = pdata.get("blocker_reason", "")
            if not blocker:
                raise ValueError("blocker_reason required for unsupported paper")
        papers[pid] = PaperEntry(
            paper_id=pdata.get("paper_id", pid),
            support_level=support_level,
            approximation_gap=pdata.get("approximation_gap", ""),
            blocker_reason=pdata.get("blocker_reason", "")
        )

    profiles = {}
    for pfid, pfdata in profiles_raw.items():
        for field, allowed in [
            ("profile_tier", PROFILE_TIERS),
            ("batching_mode", BATCHING_MODES),
            ("streaming_mode", STREAMING_MODES),
            ("stop_condition", STOP_CONDITIONS),
            ("request_order", REQUEST_ORDERS)
        ]:
            value = pfdata.get(field)
            if value not in allowed:
                raise ValueError(f"Invalid {field}: {value}")
        profiles[pfid] = ProfileEntry(
            profile_id=pfdata.get("profile_id", pfid),
            profile_tier=pfdata["profile_tier"],
            batching_mode=pfdata["batching_mode"],
            streaming_mode=pfdata["streaming_mode"],
            stop_condition=pfdata["stop_condition"],
            request_order=pfdata["request_order"]
        )

    # prompt_sources: dict[str, list[str]]
    return RegistryBundle(
        papers=papers,
        profiles=profiles,
        prompt_sources=prompt_sources
    )
