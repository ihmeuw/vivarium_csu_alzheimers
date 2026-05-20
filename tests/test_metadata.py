"""Smoke tests for project-level metadata invariants.

These guard cross-module assumptions that would otherwise blow up
mid-artifact-build with a `KeyError`.
"""

from vivarium_csu_alzheimers.constants import data_keys, data_values, metadata


def test_every_location_has_testing_rates():
    missing = [loc for loc in metadata.LOCATIONS if loc not in data_values.CSF_PET_LOCATION_TESTING_RATES]
    assert not missing, (
        f"Locations in LOCATIONS without entries in CSF_PET_LOCATION_TESTING_RATES: {missing}"
    )


def test_testing_rates_has_csf_and_pet_keys():
    for loc, rates in data_values.CSF_PET_LOCATION_TESTING_RATES.items():
        assert data_keys.TESTING_RATES.CSF in rates, f"{loc} missing CSF rate"
        assert data_keys.TESTING_RATES.PET in rates, f"{loc} missing PET rate"
