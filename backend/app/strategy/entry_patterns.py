"""Stable entry-pattern metadata helpers shared by matrix execution paths."""

from __future__ import annotations


def matched_entry_patterns(mask: int, pattern_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Decode a stable bitmask in declared order, without set/dict ordering."""
    value = int(mask)
    return tuple(pattern_id for index, pattern_id in enumerate(pattern_ids) if value & (1 << index))


def primary_entry_pattern(mask: int, pattern_ids: tuple[str, ...]) -> str | None:
    """Return the first declared matched pattern as deterministic attribution."""
    matched = matched_entry_patterns(mask, pattern_ids)
    return matched[0] if matched else None
