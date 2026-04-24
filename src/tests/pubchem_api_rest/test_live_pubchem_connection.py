"""Live integration test for the PubChem REST API.

This test performs a real network request against PubChem using a known
compound and validates the returned structural identifiers.
"""

import unittest

from src.RepoRT_data_processing.RepoRT_doublets_fixing import get_info_from_pubchem


KNOWN_COMPOUND_NAME = "alanine"
EXPECTED_INCHI = "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1"
EXPECTED_INCHIKEY = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"
EXPECTED_SMILES = "C[C@@H](C(=O)O)N"


class TestLivePubChemConnection(unittest.TestCase):
    """Live integration checks for the PubChem REST endpoint."""

    def test_known_compound_returns_expected_structure(self):
        """Verify that PubChem returns the expected structure for alanine."""
        result = get_info_from_pubchem(KNOWN_COMPOUND_NAME)
        print("Received from PubChem:", result)

        self.assertEqual(result["inchi_pubchem"], EXPECTED_INCHI)
        self.assertEqual(result["inchikey_pubchem"], EXPECTED_INCHIKEY)
        self.assertEqual(result["smiles_pubchem"], EXPECTED_SMILES)


if __name__ == "__main__":
    unittest.main()
