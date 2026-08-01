"""Resolve per-`(playerId, campaignId)` Chroma collection names."""

from __future__ import annotations

import re

_COLLECTION_PREFIX = "coach_p"
_COLLECTION_PATTERN = re.compile(r"^coach_p(\d+)_c(\d+)$")


def collection_for(player_id: int, campaign_id: int) -> str:
    """Return the Chroma collection name for `(playerId, campaignId)`.

    Both IDs must be non-negative integers; otherwise raises `ValueError`.
    """
    if player_id is None or campaign_id is None:
        raise ValueError("collection_for requires both player_id and campaign_id")

    try:
        player_int = int(player_id)
        campaign_int = int(campaign_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"collection_for requires integer ids, got player_id={player_id!r}, campaign_id={campaign_id!r}"
        ) from exc

    if player_int < 0 or campaign_int < 0:
        raise ValueError(
            f"collection_for requires non-negative ids, got player_id={player_int}, campaign_id={campaign_int}"
        )

    return f"{_COLLECTION_PREFIX}{player_int}_c{campaign_int}"


def is_coach_collection_name(name: str) -> bool:
    """Return True when `name` matches the coach collection pattern."""
    return bool(_COLLECTION_PATTERN.match(name or ""))


def parse_collection_name(name: str):
    """Return `(player_id, campaign_id)` for a coach collection name, else `None`."""
    match = _COLLECTION_PATTERN.match(name or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
