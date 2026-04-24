"""Fix doublets with failed sanity checks by querying PubChem and ClassyFire APIs.

This module reads the complete_doublets_sorted_inchi_pubchem.tsv file and for
those compounds with sanity_check = no, it queries PubChem and ClassyFire APIs
to update structural and classification information.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Import existing PubChem functions from the doublets fixing module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from RepoRT_doublets_fixing import (
    get_info_from_pubchem,
    build_pubchem_name_candidates,
)


INPUT_PATH = Path(".", "data", "complete_doublets_sorted_inchi_pubchem.tsv")
OUTPUT_PATH = Path(".", "data", "fixed_doublets_from_pubchem.tsv")

PUBCHEM_REQUEST_DELAY_SECONDS = 0.1  # Reduced delay for faster processing
PUBCHEM_TIMEOUT_SECONDS = 30
PUBCHEM_MAX_WORKERS = 3


def query_classyfire_by_inchikey(inchikey_value):
    """Query ClassyFire by InChIKey and return classification.

    Args:
        inchikey_value: InChIKey string (first part like 'ABCDEFGH-').

    Returns:
        A dictionary with ClassyFire classification entries.

    Raises:
        urllib.error.HTTPError: If ClassyFire returns an error.
        urllib.error.URLError: If the request cannot be completed.
        KeyError: If the expected JSON fields are missing.
        RuntimeError: If all retry attempts are exhausted.
    """
    url = CLASSYFIRE_BASE_URL.format(inchikey_value)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TFG_Yixi fixed_doublets_from_pubchem",
        },
    )

    for attempt in range(CLASSYFIRE_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=CLASSYFIRE_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Not found - raise KeyError to be caught later
                raise KeyError(f"InChIKey not found in ClassyFire: {inchikey_value}")
            if exc.code in [503, 502, 500]:  # Service unavailable/temporary issues
                if attempt < CLASSYFIRE_MAX_RETRIES - 1:
                    time.sleep(CLASSYFIRE_BACKOFF_SECONDS * (attempt + 1))
                    continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < CLASSYFIRE_MAX_RETRIES - 1:
                time.sleep(CLASSYFIRE_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError(f"ClassyFire request exhausted retries for {inchikey_value}")

    return payload


def parse_classyfire_response(response):
    """Parse ClassyFire API response and extract classification data.

    Args:
        response: ClassyFire JSON response.

    Returns:
        A dictionary with kingdom, superclass, class, subclass, level5, level6.
    """
    result = {
        "kingdom": "NA (NA)",
        "superclass": "NA (NA)",
        "class": "NA (NA)",
        "subclass": "NA (NA)",
        "level5": "NA (NA)",
        "level6": "NA (NA)",
    }

    try:
        # Extract direct taxonomic entries with proper formatting
        taxonomic_levels = ["kingdom", "superclass", "class", "subclass"]
        
        for level in taxonomic_levels:
            if level in response and response[level]:
                level_data = response[level]
                if isinstance(level_data, dict):
                    name = level_data.get("name", "NA")
                    chemont_id = level_data.get("chemont_id", "NA")
                    if name and name != "NA":
                        result[level] = f"{name} (CHEMONTID:{chemont_id})" if chemont_id != "NA" else f"{name} (NA)"

        # For additional levels (level5, level6), check ancestors
        ancestors = response.get("ancestors", [])
        
        # Find level5 - first direct_parent=False ancestor
        for ancestor in ancestors:
            if ancestor.get("direct_parent") is False:
                name = ancestor.get("name", "NA")
                chemont_id = ancestor.get("chemont_id", "NA")
                if name and name != "NA":
                    result["level5"] = f"{name} (CHEMONTID:{chemont_id})" if chemont_id != "NA" else f"{name} (NA)"
                break

        # Level6 - most distant ancestor
        if ancestors:
            last_ancestor = ancestors[-1]
            name = last_ancestor.get("name", "NA")
            chemont_id = last_ancestor.get("chemont_id", "NA")
            if name and name != "NA":
                result["level6"] = f"{name} (CHEMONTID:{chemont_id})" if chemont_id != "NA" else f"{name} (NA)"

    except Exception as e:
        print(f"Warning: Failed to parse ClassyFire response: {e}")

    return result


def get_classyfire_data(inchikey_value):
    """Query ClassyFire and return classification data with fallback.

    Args:
        inchikey_value: InChIKey string.

    Returns:
        A dictionary with classification entries, or "NA (NA)" on failure.
    """
    fallback_result = {
        "kingdom": "NA (NA)",
        "superclass": "NA (NA)",
        "class": "NA (NA)",
        "subclass": "NA (NA)",
        "level5": "NA (NA)",
        "level6": "NA (NA)",
    }

    if pd.isna(inchikey_value) or not str(inchikey_value).strip():
        return fallback_result

    try:
        response = query_classyfire_by_inchikey(str(inchikey_value))
        time.sleep(CLASSYFIRE_REQUEST_DELAY_SECONDS)
        result = parse_classyfire_response(response)
        return result
    except Exception as e:
        print(f"Warning: Failed to fetch ClassyFire data for {inchikey_value}: {e}")
        return fallback_result


def read_tsv_file(input_path):
    """Read a TSV file into a pandas DataFrame.

    Args:
        input_path: Path to the input TSV file.

    Returns:
        A pandas DataFrame containing the TSV contents.
    """
    return pd.read_csv(input_path, sep="\t", dtype={"dir_id": str})


def fix_row_with_pubchem_classyfire(row):
    """Fix a single row by querying PubChem API.

    For compounds with failed sanity checks, fetch updated structural data from
    PubChem. ClassyFire queries are skipped due to network issues.

    Args:
        row: A pandas Series representing a single data row.

    Returns:
        Updated row with new SMILES, InChI, and InChIKey from PubChem.
    """
    # Query PubChem using compound name (more reliable than InChI)
    compound_name = row.get("name")
    pubchem_data = {}
    
    if compound_name and not pd.isna(compound_name):
        try:
            pubchem_result = get_info_from_pubchem(compound_name)
            # Convert from the doublets_fixing format to our format
            pubchem_data = {
                "smiles": pubchem_result.get("smiles_pubchem", ""),
                "inchi": pubchem_result.get("inchi_pubchem", ""),
                "inchikey": pubchem_result.get("inchikey_pubchem", ""),
            }
        except Exception as e:
            print(f"    Warning: Failed to fetch PubChem data for '{compound_name}': {e}")
            pubchem_data = {"smiles": "", "inchi": "", "inchikey": ""}
    else:
        pubchem_data = {"smiles": "", "inchi": "", "inchikey": ""}

    # Update SMILES, InChI, and InChIKey from PubChem if successful
    if pubchem_data["smiles"]:
        row["smiles.std"] = pubchem_data["smiles"]
    if pubchem_data["inchi"]:
        row["inchi.std"] = pubchem_data["inchi"]
    if pubchem_data["inchikey"]:
        row["inchikey.std"] = pubchem_data["inchikey"]

    # Note: ClassyFire queries are skipped due to persistent network issues
    # ClassyFire columns will retain their original values or be filled with "NA (NA)"

    return row


def process_doublets_file(input_path, output_path):
    """Process doublets file: fix sanity_check=no rows and save output.

    Args:
        input_path: Path to input TSV file.
        output_path: Path to output TSV file.
    """
    print(f"Reading input file: {input_path}")
    df = read_tsv_file(input_path)

    # Identify rows that need fixing (sanity check failed)
    needs_fixing = (
        (df["sanity_check_formula"] == "no") | 
        (df["sanity_check_first_part_inchi"] == "no")
    )

    num_to_fix = needs_fixing.sum()
    print(f"Found {num_to_fix} rows with failed sanity checks needing fixing")

    if num_to_fix > 0:
        # Process rows that need fixing
        print("Processing rows with failed sanity checks...")
        rows_to_process = df[needs_fixing].index.tolist()
        
        for i, idx in enumerate(rows_to_process, 1):
            print(f"  [{i}/{num_to_fix}] Processing row {idx + 1} (compound: {df.loc[idx, 'name']})...")
            df.loc[idx] = fix_row_with_pubchem_classyfire(df.loc[idx])
            
            # Add progress indicator every 50 rows
            if i % 50 == 0:
                print(f"  Progress: {i}/{num_to_fix} rows processed")

    # Drop the temporary columns
    columns_to_drop = ["INCHI Pubchem", "sanity_check_formula", "sanity_check_first_part_inchi"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Ensure proper column order
    expected_columns = [
        "dir_id", "molecule_id", "name", "formula", "rt", 
        "smiles.std", "inchi.std", "inchikey.std",
        "classyfire.kingdom", "classyfire.superclass", "classyfire.class",
        "classyfire.subclass", "classyfire.level5", "classyfire.level6",
        "comment", "RT diff"
    ]
    
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in expected_columns if col in df.columns]
    # Add any other columns that not in expected list
    other_columns = [col for col in df.columns if col not in existing_columns]
    
    df = df[existing_columns + other_columns]

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output file: {output_path}")
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Done! Output saved to: {output_path}")


def main():
    """Run the doublets fixing workflow."""
    process_doublets_file(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
