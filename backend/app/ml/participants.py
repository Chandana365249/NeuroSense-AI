"""Participant roles under Evaluation Protocol v1.1 (docs/EVALUATION_PROTOCOL.md).

- TRUSTED_DEVELOPMENT_PARTICIPANTS: the 12 participants whose files come from the trusted local copy. They are the
  development set: all training, selection and (exploratory) cross-validation happens on them.
- FROZEN_TEST_PARTICIPANTS: the final independent test group. Development code must never load, inspect or use
  them (protocol section 6.3). The final evaluation is blocked until the data-access decision (Decision I) is made,
  and will need a separate, explicitly logged final-evaluation command.
"""

from __future__ import annotations

PROTOCOL_VERSION = "1.1"

TRUSTED_DEVELOPMENT_PARTICIPANTS: tuple[str, ...] = (
    "s01", "s02", "s03", "s05", "s07", "s08", "s09", "s11", "s15", "s16", "s22", "s25",
)

FROZEN_TEST_PARTICIPANTS: frozenset[str] = frozenset({"s12", "s17", "s18", "s20", "s21", "s23", "s29", "s30"})


def is_frozen(participant_id: str) -> bool:
    return participant_id.lower() in FROZEN_TEST_PARTICIPANTS
