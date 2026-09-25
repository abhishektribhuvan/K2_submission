"""
preprocess.py - Polars-based Text Normalization
=================================================
Memory-efficient text cleaning for business names and addresses.
Uses Polars expressions for vectorized string operations rather
than row-by-row Python loops. Handles Unicode, non-ASCII characters,
and is designed to be geographically agnostic (works for US, India,
France, or any country).
"""

import re
from pathlib import Path
import polars as pl
from typing import Optional

from src.config import (
    BUSINESS_SUFFIXES,
    ADDRESS_ABBREVIATIONS,
    POLARS_BATCH_SIZE,
)


# ============================================================================
# COMPILED REGEX PATTERNS (compiled once, reused across all calls)
# ============================================================================

# Pattern to strip non-alphanumeric characters (keep spaces)
_RE_NON_ALNUM = re.compile(r'[^a-z0-9\s]')

# Pattern to collapse multiple whitespace into single space
_RE_MULTI_SPACE = re.compile(r'\s+')

# Pattern to extract numeric tokens (street numbers, PIN codes, ZIP codes, etc.)
_RE_NUMERICS = re.compile(r'\b\d+\b')

# Compiled business suffix patterns for removal
_RE_SUFFIXES = [re.compile(pat, re.IGNORECASE) for pat in BUSINESS_SUFFIXES]

# Word boundary pattern for address abbreviation expansion
_RE_ADDR_ABBREVS = {
    re.compile(r'\b' + re.escape(abbr) + r'\b'): full
    for abbr, full in ADDRESS_ABBREVIATIONS.items()
}


import unicodedata
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def detect_and_transliterate_indic(text: str) -> str:
    """
    Ultra-fast rule-based Indic script transliteration to Romanized Latin
    plus Unicode NFKD accent normalization for European/French text.
    Supports Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada,
    Malayalam, Gurmukhi, and Oriya scripts.
    """
    if not text:
        return ""
    
    # Check if text contains non-ASCII characters
    if not any(ord(c) > 127 for c in text):
        return text

    script_counts = {}
    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:
            script_counts[sanscript.DEVANAGARI] = script_counts.get(sanscript.DEVANAGARI, 0) + 1
        elif 0x0980 <= cp <= 0x09FF:
            script_counts[sanscript.BENGALI] = script_counts.get(sanscript.BENGALI, 0) + 1
        elif 0x0A00 <= cp <= 0x0A7F:
            script_counts[sanscript.GURMUKHI] = script_counts.get(sanscript.GURMUKHI, 0) + 1
        elif 0x0A80 <= cp <= 0x0AFF:
            script_counts[sanscript.GUJARATI] = script_counts.get(sanscript.GUJARATI, 0) + 1
        elif 0x0B00 <= cp <= 0x0B7F:
            script_counts[sanscript.ORIYA] = script_counts.get(sanscript.ORIYA, 0) + 1
        elif 0x0B80 <= cp <= 0x0BFF:
            script_counts[sanscript.TAMIL] = script_counts.get(sanscript.TAMIL, 0) + 1
        elif 0x0C00 <= cp <= 0x0C7F:
            script_counts[sanscript.TELUGU] = script_counts.get(sanscript.TELUGU, 0) + 1
        elif 0x0C80 <= cp <= 0x0CFF:
            script_counts[sanscript.KANNADA] = script_counts.get(sanscript.KANNADA, 0) + 1
        elif 0x0D00 <= cp <= 0x0D7F:
            script_counts[sanscript.MALAYALAM] = script_counts.get(sanscript.MALAYALAM, 0) + 1

    if not script_counts:
        # Non-Indic unicode (e.g. French accents) -> NFKD normalization
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    try:
        primary_script = max(script_counts, key=script_counts.get)
        res = transliterate(text, primary_script, sanscript.ITRANS)
        return unicodedata.normalize('NFKD', res).encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def normalize_text(text: Optional[str]) -> str:
    """
    Core text normalization function applied to both names and addresses.

    Steps:
    1. Lowercase
    2. Transliterate Indic scripts to Romanized Latin / strip accents (NFKD)
    3. Strip punctuation (keep alphanumeric + spaces)
    4. Collapse whitespace
    5. Strip leading/trailing whitespace

    Args:
        text: Raw input string (may be None/NaN)

    Returns:
        Cleaned lowercase string, or empty string if input is null
    """
    if text is None or (isinstance(text, float)):
        return ""

    # Step 1: Convert to string & lowercase
    text = str(text).lower()

    # Step 2: Transliterate Indic scripts to Romanized Latin + strip diacritics
    text = detect_and_transliterate_indic(text)

    # Step 3: Strip punctuation, keep alphanumeric and spaces
    text = _RE_NON_ALNUM.sub(' ', text)

    # Step 4: Collapse multiple spaces
    text = _RE_MULTI_SPACE.sub(' ', text)

    # Step 5: Strip edges
    return text.strip()


def normalize_business_name(text: Optional[str]) -> str:
    """
    Normalize a business name with suffix removal.

    After base normalization, removes common legal suffixes
    (Inc, Corp, Ltd, Pvt, SARL, etc.) to focus on the distinctive
    business name tokens.

    Args:
        text: Raw business name

    Returns:
        Normalized business name with suffixes removed
    """
    text = normalize_text(text)
    if not text:
        return ""

    # Remove business legal suffixes
    for pattern in _RE_SUFFIXES:
        text = pattern.sub('', text)

    # Clean up any double spaces created by removal
    text = _RE_MULTI_SPACE.sub(' ', text).strip()
    return text


def normalize_address(text: Optional[str]) -> str:
    """
    Normalize an address with abbreviation expansion.

    Expands common address abbreviations (Rd->Road, St->Street, etc.)
    to canonical forms, enabling better fuzzy matching across sources.
    This is language-agnostic and works for US, Indian, and French addresses.

    Args:
        text: Raw business address

    Returns:
        Normalized address with expanded abbreviations
    """
    text = normalize_text(text)
    if not text:
        return ""

    # Expand address abbreviations
    for pattern, replacement in _RE_ADDR_ABBREVS.items():
        text = pattern.sub(replacement, text)

    text = _RE_MULTI_SPACE.sub(' ', text).strip()
    return text


def extract_numerics(text: Optional[str]) -> str:
    """
    Extract all numeric tokens from a string, sorted and space-joined.

    Used to capture street numbers, unit numbers, postal codes, etc.
    These numeric tokens serve as hard-match features in the classifier.
    Language-agnostic: works with US ZIP codes, Indian PINs, French postal codes.

    Args:
        text: Normalized text string

    Returns:
        Space-separated sorted numeric tokens, or empty string
    """
    if not text:
        return ""
    nums = _RE_NUMERICS.findall(text)
    return " ".join(sorted(set(nums)))


def create_combined_field(name: str, address: str) -> str:
    """
    Create a combined name+address field for TF-IDF blocking.

    Concatenates normalized name and address for the blocking index,
    giving the TF-IDF vectorizer both signals simultaneously.

    Args:
        name: Normalized business name
        address: Normalized business address

    Returns:
        Combined "name address" string
    """
    parts = []
    if name:
        parts.append(name)
    if address:
        parts.append(address)
    return " ".join(parts)


def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply full preprocessing pipeline to a Polars DataFrame.

    Adds the following columns:
    - name_clean: Normalized business name (suffixes removed)
    - addr_clean: Normalized address (abbreviations expanded)
    - name_addr_combined: Combined field for TF-IDF blocking
    - name_numerics: Numeric tokens from name
    - addr_numerics: Numeric tokens from address

    Args:
        df: Input DataFrame with columns [entity_id, business_name,
            business_address, country]

    Returns:
        DataFrame with additional normalized columns
    """
    # Apply normalization using map_elements (vectorized Python UDFs)
    # This is faster than iterating rows in pure Python
    df = df.with_columns([
        # Normalize business name
        pl.col("business_name").map_elements(
            normalize_business_name, return_dtype=pl.Utf8
        ).alias("name_clean"),

        # Normalize address
        pl.col("business_address").map_elements(
            normalize_address, return_dtype=pl.Utf8
        ).alias("addr_clean"),

        # Lowercase country for consistent grouping
        pl.col("country").map_elements(
            lambda x: str(x).strip().lower() if x else "unknown",
            return_dtype=pl.Utf8
        ).alias("country_clean"),
    ])

    # Create combined field and extract numerics
    df = df.with_columns([
        # Combined name + address for blocking
        pl.struct(["name_clean", "addr_clean"]).map_elements(
            lambda row: create_combined_field(row["name_clean"], row["addr_clean"]),
            return_dtype=pl.Utf8
        ).alias("name_addr_combined"),

        # Extract numeric tokens from name
        pl.col("name_clean").map_elements(
            extract_numerics, return_dtype=pl.Utf8
        ).alias("name_numerics"),

        # Extract numeric tokens from address
        pl.col("addr_clean").map_elements(
            extract_numerics, return_dtype=pl.Utf8
        ).alias("addr_numerics"),
    ])

    return df


def load_and_preprocess(filepath: str, batch_size: int = POLARS_BATCH_SIZE) -> pl.DataFrame:
    """
    Load a TSV file and preprocess, using Parquet disk caching to avoid
    re-cleaning text on subsequent pipeline runs.

    Args:
        filepath: Path to the .tsv file
        batch_size: Number of rows per batch

    Returns:
        Fully preprocessed Polars DataFrame
    """
    from src.config import CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    path_obj = Path(filepath)
    cache_path = CACHE_DIR / f"{path_obj.stem}_preprocessed.parquet"

    if cache_path.exists():
        print(f"[PREPROCESS] Loading cached preprocessed data: {cache_path}")
        return pl.read_parquet(cache_path)

    print(f"[PREPROCESS] Loading and preprocessing: {filepath}")

    # Read the full file with Polars
    df = pl.read_csv(
        filepath,
        separator="\t",
        has_header=True,
        null_values=["", "NA", "null", "None"],
        dtypes={
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8,
        },
        ignore_errors=True,
    )

    print(f"[PREPROCESS]   Loaded {len(df):,} rows")

    # Fill nulls with empty strings to prevent downstream errors
    df = df.with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("unknown"),
    ])

    # Process in batches to control memory usage
    chunks = []
    total_rows = len(df)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk = df.slice(start, end - start)
        chunk = preprocess_dataframe(chunk)
        chunks.append(chunk)

        if (start // batch_size) % 5 == 0:
            print(f"[PREPROCESS]   Processed {end:,}/{total_rows:,} rows")

    result = pl.concat(chunks)
    print(f"[PREPROCESS]   Preprocessing complete: {len(result):,} rows")

    # Cache preprocessed dataframe
    try:
        result.write_parquet(cache_path, compression="snappy")
        print(f"[PREPROCESS]   Cached preprocessed data to: {cache_path}")
    except Exception as e:
        print(f"[PREPROCESS]   Warning: Failed to write cache: {e}")

    return result


def load_ground_truth(filepath: str) -> dict:
    """
    Load ground truth file into a dictionary mapping.

    Returns a dict where keys are Source 1 entity IDs and values are
    sets of matched entity IDs (from Source 2 and Source 3).
    Singletons (no matches) map to empty sets.

    Args:
        filepath: Path to train_ground_truth.tsv

    Returns:
        Dict[str, Set[str]] mapping source1_id -> {matched_ids}
    """
    print(f"[PREPROCESS] Loading ground truth: {filepath}")

    df = pl.read_csv(
        filepath,
        separator="\t",
        has_header=True,
        null_values=[""],
        dtypes={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        ignore_errors=True,
    )

    gt = {}
    for row in df.iter_rows(named=True):
        s1_id = row["source1_entity_id"].strip()
        matched = row.get("matched_entity_ids")
        if matched and str(matched).strip():
            # Split comma-separated IDs, strip whitespace, deduplicate
            ids = set(m.strip() for m in str(matched).split(",") if m.strip())
        else:
            ids = set()
        gt[s1_id] = ids

    print(f"[PREPROCESS]   Loaded {len(gt):,} ground truth entries")
    n_singletons = sum(1 for v in gt.values() if len(v) == 0)
    print(f"[PREPROCESS]   Singletons (no matches): {n_singletons:,}")
    print(f"[PREPROCESS]   With matches: {len(gt) - n_singletons:,}")

    return gt


if __name__ == "__main__":
    # Quick test of preprocessing on a small sample
    from src.config import TRAIN_SOURCE1, TRAIN_GROUND_TRUTH

    # Load first 1000 rows of source1 for testing
    df = pl.read_csv(
        str(TRAIN_SOURCE1),
        separator="\t",
        has_header=True,
        n_rows=1000,
        null_values=[""],
    )
    df = df.with_columns([
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("unknown"),
    ])

    result = preprocess_dataframe(df)
    print("\nSample preprocessed data:")
    print(result.head(5))
    print(f"\nColumns: {result.columns}")
