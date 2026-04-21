"""Unit tests for the PubChem REST helper functions.

These tests cover only the PubChem-related functionality from
``RepoRT_doublets_fixing.py`` and intentionally avoid the rest of the
doublets-processing pipeline.
"""

import json
import unittest

from unittest.mock import patch

from src.RepoRT_data_processing.RepoRT_doublets_fixing import (
    build_pubchem_name_candidates,
    get_info_from_pubchem,
    request_pubchem_properties,
)


class MockHttpResponse:
    """Minimal HTTP response mock for ``urllib.request.urlopen`` tests."""

    def __init__(self, payload):
        """Store the encoded payload returned by ``read()``."""
        self._payload = payload

    def read(self):
        """Return the encoded response body."""
        return self._payload

    def __enter__(self):
        """Support context-manager usage."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Support context-manager usage."""
        return False


class TestPubChemApiRest(unittest.TestCase):
    """Tests for the PubChem REST helper functions."""

    def test_build_pubchem_name_candidates_keeps_original_and_strips_star(self):
        """Return both the original name and the star-stripped fallback."""
        result = build_pubchem_name_candidates("alanine*")
        self.assertEqual(result, ["alanine*", "alanine"])

    @patch("urllib.request.urlopen")
    def test_request_pubchem_properties_parses_expected_fields(self, mock_urlopen):
        """Parse InChI, InChIKey, and SMILES from a PubChem JSON response."""
        payload = {
            "PropertyTable": {
                "Properties": [
                    {
                        "InChI": "InChI=1S/C3H7NO2/test",
                        "InChIKey": "QNAYBMKLOCPYGJ-UHFFFAOYSA-N",
                        "IsomericSMILES": "CC(C(=O)O)N",
                    }
                ]
            }
        }
        mock_urlopen.return_value = MockHttpResponse(json.dumps(payload).encode("utf-8"))

        result = request_pubchem_properties("alanine")

        self.assertEqual(result["inchi_pubchem"], "InChI=1S/C3H7NO2/test")
        self.assertEqual(result["inchikey_pubchem"], "QNAYBMKLOCPYGJ-UHFFFAOYSA-N")
        self.assertEqual(result["smiles_pubchem"], "CC(C(=O)O)N")

    @patch("src.RepoRT_data_processing.RepoRT_doublets_fixing.request_pubchem_properties")
    def test_get_info_from_pubchem_uses_fallback_name(self, mock_request_pubchem_properties):
        """Retry with a normalized name when the original name fails."""
        mock_request_pubchem_properties.side_effect = [
            KeyError("not found"),
            {
                "inchi_pubchem": "InChI=1S/C3H7NO2/test",
                "inchikey_pubchem": "QNAYBMKLOCPYGJ-UHFFFAOYSA-N",
                "smiles_pubchem": "CC(C(=O)O)N",
            },
        ]

        result = get_info_from_pubchem("alanine*")

        self.assertEqual(result["inchi_pubchem"], "InChI=1S/C3H7NO2/test")
        self.assertEqual(mock_request_pubchem_properties.call_count, 2)

    @patch("src.RepoRT_data_processing.RepoRT_doublets_fixing.request_pubchem_properties")
    def test_get_info_from_pubchem_returns_empty_values_when_lookup_fails(self, mock_request_pubchem_properties):
        """Return empty PubChem fields when all lookup candidates fail."""
        mock_request_pubchem_properties.side_effect = KeyError("not found")

        result = get_info_from_pubchem("unknown_compound")

        self.assertEqual(
            result,
            {
                "inchi_pubchem": "",
                "inchikey_pubchem": "",
                "smiles_pubchem": "",
            },
        )


if __name__ == "__main__":
    unittest.main()
