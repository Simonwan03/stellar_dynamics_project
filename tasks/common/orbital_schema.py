"""Shared field order and perturbation-pair conventions for task packages."""

FULL_RESULT_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "masses_complete",
    "eccs_complete",
    "smaxes_complete",
    "energy_complete",
    "angMomentum_complete",
    "APs_complete",
    "LANs_complete",
    "incs_complete",
)
ORBITAL15_RESULT_KEYS = (
    "aratios_complete",
    "mratios_complete",
    "eccs_complete",
    "incs_complete",
    "APs_complete",
    "LANs_complete",
)
ANGLE_RESULT_KEYS = {"APs_complete", "LANs_complete"}

DEFAULT_VALID_PAIRS = tuple((i, i + 16) for i in range(16))

INITIAL_FEATURE_KEYS = (
    "masses_complete",
    "mratios_complete",
    "smaxes_complete",
    "eccs_complete",
    "incs_complete",
    "aratios_complete",
    "energy_complete",
    "angMomentum_complete",
    "periods_complete",
)
