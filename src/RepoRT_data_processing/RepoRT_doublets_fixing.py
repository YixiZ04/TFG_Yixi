"""Utilities for sorting and enriching RepoRT doublets tables.

This module reads a doublets TSV file, sorts it by ``dir_id`` and
``smiles.std``, retrieves structural information from PubChem using the
compound name, applies structural sanity checks, computes RT differences for
validated doublets, and writes the resulting TSV files.
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


DEFAULT_INPUT_CANDIDATES = [
    Path(".", "data", "complete_doublets.tsv"),
    Path(".", "data", "doublets", "complete_doublets.tsv"),
]
SORTED_OUTPUT_PATH = Path(".", "data", "complete_doublets_sorted.tsv")
ENRICHED_OUTPUT_PATH = Path(".", "data", "complete_doublets_sorted_inchi_pubchem.tsv")
PUBCHEM_BASE_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "InChI,InChIKey,IsomericSMILES/JSON"
)
PUBCHEM_REQUEST_DELAY_SECONDS = 1
PUBCHEM_TIMEOUT_SECONDS = 30
PUBCHEM_MAX_WORKERS = 1
PUBCHEM_MAX_RETRIES = 5
PUBCHEM_BACKOFF_SECONDS = 1.5


def resolve_input_path(candidate_paths=DEFAULT_INPUT_CANDIDATES):
    """Return the first existing input path from a list of candidates.

    Args:
        candidate_paths: Iterable of candidate ``Path`` objects to inspect.

    Returns:
        The first existing input path.

    Raises:
        FileNotFoundError: If none of the candidate paths exists.
    """
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Input file not found. Expected one of: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def read_tsv_file(input_path):
    """Read a TSV file into a pandas DataFrame.

    Args:
        input_path: Path to the input TSV file.

    Returns:
        A pandas DataFrame containing the TSV contents.
    """
    return pd.read_csv(input_path, sep="\t", dtype={"dir_id": str})


def sort_doublets_dataframe(input_df):
    """Sort the doublets DataFrame by the requested tuple.

    Args:
        input_df: Input DataFrame containing the doublets table.

    Returns:
        A new DataFrame sorted by ``dir_id`` and ``smiles.std`` while keeping
        full rows aligned.
    """
    return input_df.sort_values(
        by=["dir_id", "smiles.std", "name", "molecule_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_pubchem_name_candidates(compound_name):
    """Build a short list of PubChem lookup candidates for a compound name.

    Args:
        compound_name: Compound name stored in the input table.

    Returns:
        A list of candidate names to try against PubChem.
    """
    if pd.isna(compound_name):
        return []

    cleaned_name = str(compound_name).strip()
    if not cleaned_name:
        return []

    candidates = [cleaned_name]
    stripped_star = cleaned_name.rstrip("*").strip()
    if stripped_star and stripped_star not in candidates:
        candidates.append(stripped_star)

    return candidates


def request_pubchem_properties(compound_name):
    """Query PubChem by compound name and return structural properties.

    Args:
        compound_name: Compound name to search in PubChem.

    Returns:
        A dictionary with PubChem InChI, InChIKey, and isomeric SMILES.

    Raises:
        urllib.error.HTTPError: If PubChem returns a non-recoverable HTTP error.
        urllib.error.URLError: If the request cannot be completed.
        KeyError: If the expected JSON fields are missing.
        IndexError: If PubChem returns an empty property list.
        RuntimeError: If all retry attempts are exhausted.
    """
    encoded_name = urllib.parse.quote(compound_name, safe="")
    url = PUBCHEM_BASE_URL.format(name=encoded_name)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "TFG_Yixi RepoRT doublets fixer",
        },
    )

    for attempt in range(PUBCHEM_MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=PUBCHEM_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 503 or attempt == PUBCHEM_MAX_RETRIES - 1:
                raise
            time.sleep(PUBCHEM_BACKOFF_SECONDS * (attempt + 1))
    else:
        raise RuntimeError(f"PubChem request failed for {compound_name}")

    properties = payload["PropertyTable"]["Properties"][0]
    return {
        "inchi_pubchem": properties.get("InChI", ""),
        "inchikey_pubchem": properties.get("InChIKey", ""),
        "smiles_pubchem": properties.get("SMILES", ""),
    }


def get_info_from_pubchem(compound_name):
    """Return PubChem structural data for one compound name.

    This function applies light name normalization and falls back to empty
    values when PubChem does not return a usable result.

    Args:
        compound_name: Compound name stored in the input table.

    Returns:
        A dictionary with PubChem InChI, InChIKey, and isomeric SMILES.
    """
    fallback_result = {
        "inchi_pubchem": "",
        "inchikey_pubchem": "",
        "smiles_pubchem": "",
    }

    for candidate_name in build_pubchem_name_candidates(compound_name):
        try:
            result = request_pubchem_properties(candidate_name)
            time.sleep(PUBCHEM_REQUEST_DELAY_SECONDS)
            return result
        except (KeyError, IndexError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue

    return fallback_result


def fetch_pubchem_data_for_names(compound_names):
    """Query PubChem once per unique compound name.

    Args:
        compound_names: Iterable of compound names from the input table.

    Returns:
        A dictionary mapping each unique compound name to its PubChem
        structural data.
    """
    unique_names = [name for name in pd.Series(compound_names).dropna().unique()]
    cache = {}

    with ThreadPoolExecutor(max_workers=PUBCHEM_MAX_WORKERS) as executor:
        future_to_name = {
            executor.submit(get_info_from_pubchem, compound_name): compound_name
            for compound_name in unique_names
        }
        for future in as_completed(future_to_name):
            compound_name = future_to_name[future]
            try:
                cache[compound_name] = future.result()
            except Exception:
                cache[compound_name] = {
                    "inchi_pubchem": "",
                    "inchikey_pubchem": "",
                    "smiles_pubchem": "",
                }

    return cache


def extract_formula_from_inchi(inchi_value):
    """Extract the molecular formula section from an InChI string.

    Args:
        inchi_value: InChI string to parse.

    Returns:
        The molecular formula if it can be extracted, otherwise an empty
        string.
    """
    if pd.isna(inchi_value):
        return ""

    inchi_text = str(inchi_value).strip()
    if not inchi_text or not inchi_text.startswith("InChI="):
        return ""

    parts = inchi_text.split("/")
    if len(parts) < 2:
        return ""

    return parts[1].strip()


def sanity_check_formula(inchi_1, inchi_2):
    """Check whether two InChI strings share the same molecular formula.

    Args:
        inchi_1: First InChI string.
        inchi_2: Second InChI string.

    Returns:
        ``"yes"`` when both formulas are present and equal, otherwise ``"no"``.
    """
    formula_1 = extract_formula_from_inchi(inchi_1)
    formula_2 = extract_formula_from_inchi(inchi_2)
    return "yes" if formula_1 and formula_1 == formula_2 else "no"


def extract_first_part_inchikey(inchikey_value):
    """Extract the first block of an InChIKey.

    Args:
        inchikey_value: InChIKey string to parse.

    Returns:
        The first InChIKey block, or an empty string when unavailable.
    """
    if pd.isna(inchikey_value):
        return ""

    inchikey_text = str(inchikey_value).strip()
    if not inchikey_text:
        return ""

    return inchikey_text.split("-")[0]


def sanity_check_inchi_first_part(inchikey_1, inchikey_2):
    """Compare the first block of two InChIKeys.

    Args:
        inchikey_1: First InChIKey string.
        inchikey_2: Second InChIKey string.

    Returns:
        ``"yes"`` when both first blocks are present and equal, otherwise
        ``"no"``.
    """
    first_part_1 = extract_first_part_inchikey(inchikey_1)
    first_part_2 = extract_first_part_inchikey(inchikey_2)
    return "yes" if first_part_1 and first_part_1 == first_part_2 else "no"


def insert_column_next_to(df, reference_column, new_column_name, values):
    """Insert a new column immediately after a reference column.

    Args:
        df: Source DataFrame.
        reference_column: Existing column after which the new column is placed.
        new_column_name: Name of the inserted column.
        values: Column values to insert.

    Returns:
        A copy of the input DataFrame with the inserted column.
    """
    result_df = df.copy()
    insert_position = result_df.columns.get_loc(reference_column) + 1
    result_df.insert(insert_position, new_column_name, values)
    return result_df


def enrich_with_pubchem_data(sorted_df):
    """Add PubChem-derived structural columns and sanity checks.

    Args:
        sorted_df: Sorted doublets DataFrame.

    Returns:
        A copy of the DataFrame containing ``INCHI Pubchem``,
        ``sanity_check_formula``, and ``sanity_check_first_part_inchi``.
    """
    cache = fetch_pubchem_data_for_names(sorted_df["name"])
    pubchem_rows = sorted_df["name"].apply(
        lambda compound_name: cache.get(
            compound_name,
            {"inchi_pubchem": "", "inchikey_pubchem": "", "smiles_pubchem": ""},
        )
    )
    inchi_pubchem_values = pubchem_rows.apply(lambda row: row["inchi_pubchem"])
    inchikey_pubchem_values = pubchem_rows.apply(lambda row: row["inchikey_pubchem"])

    enriched_df = insert_column_next_to(
        sorted_df,
        "inchi.std",
        "INCHI Pubchem",
        inchi_pubchem_values,
    )

    inchi_pubchem_position = enriched_df.columns.get_loc("INCHI Pubchem")
    enriched_df.insert(
        inchi_pubchem_position + 1,
        "sanity_check_formula",
        [
            sanity_check_formula(local_inchi, pubchem_inchi)
            for local_inchi, pubchem_inchi in zip(enriched_df["inchi.std"], enriched_df["INCHI Pubchem"])
        ],
    )
    enriched_df.insert(
        inchi_pubchem_position + 2,
        "sanity_check_first_part_inchi",
        [
            sanity_check_inchi_first_part(local_inchikey, pubchem_inchikey)
            for local_inchikey, pubchem_inchikey in zip(enriched_df["inchikey.std"], inchikey_pubchem_values)
        ],
    )

    return enriched_df


def calculate_rt_diff(enriched_df):
    """Compute RT differences for validated doublet pairs.

    RT differences are written only for groups that contain exactly two rows and
    where both sanity checks are ``"yes"`` for both rows.

    Args:
        enriched_df: Enriched doublets DataFrame.

    Returns:
        A copy of the DataFrame with the ``RT diff`` column populated when the
        pair is validated.
    """
    result_df = enriched_df.copy()
    result_df["RT diff"] = pd.NA

    grouping_columns = ["dir_id", "smiles.std"]
    for _, group_df in result_df.groupby(grouping_columns, sort=False):
        valid_group_df = group_df[
            (group_df["sanity_check_formula"] == "yes")
            & (group_df["sanity_check_first_part_inchi"] == "yes")
        ]

        if len(group_df) == 2 and len(valid_group_df) == 2:
            rt_difference = abs(group_df.iloc[0]["rt"] - group_df.iloc[1]["rt"])
            result_df.loc[group_df.index, "RT diff"] = rt_difference

    return result_df


def write_results_tsv(dataframe, output_path):
    """Write a DataFrame to a TSV file.

    Args:
        dataframe: DataFrame to export.
        output_path: Destination path for the TSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", index=False)


def main():
    """Run the full RepoRT doublets processing workflow."""
    input_path = resolve_input_path()
    input_df = read_tsv_file(input_path)

    sorted_df = sort_doublets_dataframe(input_df)
    write_results_tsv(sorted_df, SORTED_OUTPUT_PATH)

    enriched_df = enrich_with_pubchem_data(sorted_df)
    final_df = calculate_rt_diff(enriched_df)
    write_results_tsv(final_df, ENRICHED_OUTPUT_PATH)

    print(f"Input file used: {input_path}")
    print(f"Sorted output written to: {SORTED_OUTPUT_PATH}")
    print(f"Enriched output written to: {ENRICHED_OUTPUT_PATH}")


def has_failed_sanity_checks(row):
    """Check if a row has any failed sanity checks.

    Args:
        row: A pandas Series representing a data row.

    Returns:
        True if sanity_check_formula or sanity_check_first_part_inchi is "no".
    """
    return (
        row.get("sanity_check_formula") == "no" or
        row.get("sanity_check_first_part_inchi") == "no"
    )


def get_pubchem_data_with_synonyms(compound_name):
    """Get PubChem data for a compound name, trying synonyms if needed.

    Args:
        compound_name: Compound name to search in PubChem.

    Returns:
        A dictionary with PubChem InChI, InChIKey, and isomeric SMILES.
        Returns empty dict if not found.
    """
    if pd.isna(compound_name) or not str(compound_name).strip():
        return {}

    # First try the compound name directly
    try:
        result = request_pubchem_properties(str(compound_name))
        time.sleep(PUBCHEM_REQUEST_DELAY_SECONDS)
        return {
            "inchi": result.get("InChI", ""),
            "inchikey": result.get("InChIKey", ""),
            "smiles": result.get("SMILES", ""),
        }
    except (KeyError, IndexError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        pass

    # If direct name fails, try to get synonyms and search those
    try:
        synonyms = get_pubchem_synonyms(str(compound_name))
        for synonym in synonyms[:3]:  # Try up to 3 synonyms
            try:
                result = request_pubchem_properties(synonym)
                time.sleep(PUBCHEM_REQUEST_DELAY_SECONDS)
                return {
                    "inchi": result.get("InChI", ""),
                    "inchikey": result.get("InChIKey", ""),
                    "smiles": result.get("SMILES", ""),
                }
            except (KeyError, IndexError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
    except Exception:
        pass

    return {}


def get_pubchem_synonyms(compound_name):
    """Get synonyms for a compound from PubChem.

    Args:
        compound_name: Compound name to get synonyms for.

    Returns:
        A list of synonym strings.
    """
    encoded_name = urllib.parse.quote(compound_name, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/synonyms/JSON"

    for attempt in range(PUBCHEM_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=PUBCHEM_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 503 or attempt == PUBCHEM_MAX_RETRIES - 1:
                return []
            time.sleep(PUBCHEM_BACKOFF_SECONDS * (attempt + 1))
    else:
        return []

    try:
        synonyms = payload["InformationList"]["Information"][0]["Synonym"]
        return [s for s in synonyms if s != compound_name][:10]  # Return up to 10 synonyms, excluding original name
    except (KeyError, IndexError):
        return []


def update_row_from_pubchem(row, pubchem_data):
    """Update a row with data from PubChem.

    Args:
        row: A pandas Series representing a data row.
        pubchem_data: Dictionary with PubChem data.

    Returns:
        Updated row Series.
    """
    updated_row = row.copy()

    if pubchem_data.get("smiles"):
        updated_row["smiles.std"] = pubchem_data["smiles"]
    if pubchem_data.get("inchi"):
        updated_row["inchi.std"] = pubchem_data["inchi"]
    if pubchem_data.get("inchikey"):
        updated_row["inchikey.std"] = pubchem_data["inchikey"]

    return updated_row


def test_fix_doublets_first_two_lines():
    """Unit test: Process first two lines from complete_doublets_sorted_inchi_pubchem.tsv.

    Only processes lines where sanity checks failed, gets PubChem data with synonyms,
    updates the rows, and writes to fixed_doublets.tsv.
    """
    input_path = Path(".", "data", "complete_doublets_sorted_inchi_pubchem.tsv")

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    # Read the input file
    df = read_tsv_file(input_path)

    # Process only the first two rows
    rows_to_process = df.head(2)
    updated_rows = []

    for idx, row in rows_to_process.iterrows():
        print(f"Processing row {idx + 1}: {row['name']}")

        if has_failed_sanity_checks(row):
            print(f"  Row has failed sanity checks. Getting PubChem data...")

            pubchem_data = get_pubchem_data_with_synonyms(row["name"])

            if pubchem_data:
                print(f"  Found PubChem data: SMILES={pubchem_data.get('smiles', '')[:30]}...")
                updated_row = update_row_from_pubchem(row, pubchem_data)
                updated_rows.append(updated_row)
                print("  Row updated successfully")
            else:
                print("  No PubChem data found, keeping original row")
                updated_rows.append(row)
        else:
            print("  Row passed sanity checks, keeping as-is")
            updated_rows.append(row)

    # Create output DataFrame
    output_df = pd.DataFrame(updated_rows)

    # Remove the sanity check columns as requested
    columns_to_drop = ["INCHI Pubchem", "sanity_check_formula", "sanity_check_first_part_inchi"]
    output_df = output_df.drop(columns=[col for col in columns_to_drop if col in output_df.columns])

    # Write to fixed_doublets.tsv
    output_path = Path(".", "data", "fixed_doublets.tsv")
    write_results_tsv(output_df, output_path)

    print(f"Test completed. Output written to: {output_path}")
    print(f"Processed {len(rows_to_process)} rows, {len(updated_rows)} rows in output")


if __name__ == "__main__":
    # Uncomment the line below to run the unit test instead of main()
    test_fix_doublets_first_two_lines()
    # main()
