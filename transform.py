import calendar
import numpy as np
import pandas as pd
import pycountry
import re
import unicodedata
from functools import lru_cache
from typing import SupportsInt, Iterable


# ID
ID_COL = "unique_id"

# Names
LATIN_NAME_COLS = [f"name_{i}" for i in range(1, 7)]
NONLATIN_NAME_COL = "name_nonlatin_script"
FULL_NAME_COL = "full_name"
NORMALISED_FULL_NAME_COL = "normalised_full_name"

# D.O.B
DOB_COL = "dob"
DOB_YEAR_COL = "dob_year"
DOB_MONTH_COL = "dob_month"
DOB_DAY_COL = "dob_day"


def restructure_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw CSV was found to be read in as a dataframe with 1 column called
    "Report Date: 29-Apr-2026". The series under this singular column
    contains the header and rows for the restructured dataframe built here.

    The pandas dataframe created from "UK-Sanctions-List.csv" has one
    column ("Report Date: 29-Apr-2026"), whose values contain the final
    field of each row, while the preceding fields became the index.
    """
    # extract the only column as a series
    s = df[df.columns[0]]
    # rebuild rows
    rows = [
        [
            *idx,   # first to penultimate field
            val     # final field
        ]
        for idx, val in zip(s.index, s.values)
    ]
    return pd.DataFrame(rows[1:], columns=rows[0])


@lru_cache
def normalise_label(s: str) -> str:
    """Normalise column names (some names call for further tidying)."""
    s = s.strip().lower()
    # replace spaces/slashes with underscores
    s = re.sub(r"[ /]+", "_", s)
    # remove punctuation except underscores
    s = re.sub(r"[^a-z0-9_]", "", s)
    # collapse repeated underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalise_categorical_data(
    df: pd.DataFrame,
    data: dict[str, Iterable[str]]
) -> pd.DataFrame:
    """Normalise data whose meaning is not altered via normalisation.
    `data` has items like column_name: expected_values."""
    df = df.copy()

    for col in data.keys():
        if col in df.columns:
            # ignore NaNs
            mask = df[col].notna()
            # normalise data
            df.loc[mask, col] = df.loc[mask, col].apply(normalise_label)

    # check for unexpected values
    for col, expected in data.items():
        # drop NaNs and select values who aren't expected
        unexpected = (
            df[col]
            .dropna()
            [~df[col].dropna().isin(expected)]
            .unique()
        )
        
        if len(unexpected) > 0:
            print(f"[WARNING] Unexpected values in {col}: {unexpected}")

    return df


# ===========================================================================
# NAME PROCESSING
# ===========================================================================

def build_full_name(row) -> str:
    """Build full_name from names present in row. Assumes
    all columns of row are name columns to be joined, in order."""
    return " ".join(row.dropna().astype(str)).strip()


@lru_cache
def remove_accents(name: str) -> str:
    """Used to normalise names."""
    if not unicodedata.combining(c):
        return np.nan
    return "".join([c for c in unicodedata.normalize("NFKD", name)])


@lru_cache
def replace_punctuation_with_whitespace(name: str) -> str:
    """Used to normalise names."""
    return re.sub(r"[^\w\s]", " ", name)


@lru_cache
def normalise_whitespace(s: str) -> str:
    """Used to normalise names."""
    return re.sub(r"\s+", " ", s)


@lru_cache
def normalise_name(name: str) -> str:
    """Normalise names for matching purposes, provided exclusively
    in addition to their raw counterparts."""
    if pd.isnull(name):
        return np.nan

    text = unicodedata.normalize("NFKC", name)
    text = text.upper()
    text = remove_accents(text)
    text = replace_punctuation_with_whitespace(text)
    text = normalise_whitespace(text)

    return text.strip()


@lru_cache
def remove_whitespace(name: str) -> str:
    """Used to construct compact names, another type of name normalisation."""
    return re.sub(r"\s+", "", name)


@lru_cache
def compact_name(name: str) -> str:
    """Like normalised names, compact names are used for matching purposes,
    provided exclusively in addition to their raw counterparts."""
    if pd.isnull(name):
        return np.nan
    return remove_whitespace(name)


def remove_least_complete_names(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce name dataframe by selecting
    most complete name where applicable."""

    df = df.copy()

    # add precision_score to help with D.O.B selection
    df["precision_score"] = (
        df[LATIN_NAME_COLS]
        .notna()
        .sum(axis=1)
    )

    # sort so most complete names come first
    df = (
        df.sort_values(
            ["unique_id", "precision_score"],
            ascending=[True, False]
        )
    )

    rows_to_keep = []

    for uid, group in df.groupby(["unique_id", "full_name"]):

        kept = []

        for idx, row in group.iterrows():

            is_redundant = False

            for kept_row in kept:

                compatible = True

                for col in LATIN_NAME_COLS:

                    current = row[col]
                    existing = kept_row[col]

                    if pd.notna(current) and pd.notna(existing):
                        if current != existing:
                            compatible = False
                            break
                
                if compatible:
                    is_redundant = True
                    break
            
            if not is_redundant:
                kept.append(row)
                rows_to_keep.append(idx)

    df = (
        df.loc[rows_to_keep]
        .drop(columns=["precision_score"])
        .sort_index()
    )

    return df


# ===========================================================================
# DATE PROCESSING
# ===========================================================================

@lru_cache
def valid_year(year: SupportsInt) -> bool:
    # fine to hard-code 2026 iff this script is only utilised
    # for the specific dataset in question
    # return 1900 <= int(year) <= 2026
    return int(year) <= 2026


@lru_cache
def valid_month(month: SupportsInt) -> bool:
    return 1<= int(month) <= 12


@lru_cache
def valid_day(day: SupportsInt, month: SupportsInt, year: SupportsInt) -> bool:
    try:
        max_day = calendar.monthrange(int(year), int(month))[1]
        return 1 <= int(day) <= max_day
    except:
        return False


@lru_cache
def parse_dob(raw_dob):
    """Used to parse D.O.Bs with inconsistent format, missing
    values, or have placeholder values. Extracts year, month,
    and day (where possible), and records precision and validity."""

    result = {
        "raw_dob": raw_dob,
        "dob_year": None,
        "dob_month": None,
        "dob_day": None,
        "dob_precision": None,
        "dob_valid_flag": False,
        "dob_parse_issues": []
    }

    if pd.isnull(raw_dob):
        result["dob_parse_issues"].append("missing_dob")
        return result

    raw = str(raw_dob).strip()

    # split date components
    parts = raw.split("/")

    # YYYY
    if re.fullmatch(r"\d{4}", raw):
        
        if valid_year(raw):
            result.update({
                "dob_year": raw,
                "dob_precision": "year_only",
                "dob_valid_flag": True
            })
        
        else:
            result["dob_parse_issues"].append("invalid_year")
        
        return result

    if len(parts) != 3:
        result["dob_parse_issues"].append("unrecognized_format")
        return result

    day, month, year = parts

    # ??/??/YYYY
    if year.isdigit():
        
        if valid_year(year):
            result["dob_year"] = year
        
        else:
            result["dob_parse_issues"].append("invalid_year")
            return result

    else:
        result["dob_parse_issues"].append("non_numeric_year")
        return result

    has_year = result["dob_year"] is not None

    # ??/MM/yyyy
    if month.isdigit():
        
        if valid_month(month):
            result["dob_month"] = f"{int(month):0>2}"

        else:
            result["dob_parse_issues"].append("invalid_month")

    else:
        result["dob_parse_issues"].append("non_numeric_month")

    has_month = result["dob_month"] is not None

    # DD/mm/yyyy
    if day.isdigit():

        if has_month:

            if valid_day(day, result["dob_month"], result["dob_year"]):
                result["dob_day"] = f"{int(day):0>2}"

            else:
                result["dob_parse_issues"].append("invalid_day")

        else:
            result["dob_parse_issues"].append("day_without_valid_month")

    else:
        result["dob_parse_issues"].append("non_numeric_day") ###

    has_day = result["dob_day"] is not None

    # Determine precision
    if has_year and has_month and has_day:
        result["dob_precision"] = "exact"

    elif has_year and has_month:
        result["dob_precision"] = "month_year"

    elif has_year:
        result["dob_precision"] = "year_only"

    result["dob_valid_flag"] = has_year

    return result


def remove_least_complete_dobs(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce D.O.B dataframe by selecting
    most complete D.O.B where applicable."""

    df = df.copy()

    # add precision_score to help with D.O.B selection
    df["precision_score"] = (
        df[["year", "month", "day"]]
        .notna()
        .sum(axis=1)
    )

    # sort so most complete D.O.Bs come first
    df = (
        df.sort_values(
            ["unique_id", "precision_score"],
            ascending=[True, False]
        )
    )

    rows_to_keep = []

    for uid, group in df.groupby("unique_id"):
        
        kept = []

        for idx, row in group.iterrows():

            is_redundant = False

            for kept_row in kept:

                compatible = True

                for col in ["year", "month", "day"]:

                    current = row[col]
                    existing = kept_row[col]

                    if pd.notna(current) and pd.notna(existing):
                        if current != existing:
                            compatible = False
                            break
                
                if compatible:
                    is_redundant = True
                    break
            
            if not is_redundant:
                kept.append(row)
                rows_to_keep.append(idx)

    df = (
        df.loc[rows_to_keep]
        .drop(columns=["precision_score"])
        .sort_index()
    )

    return df


# ===========================================================================
# COUNTRY PROCESSING
# ===========================================================================

@lru_cache
def normalise_country(country: str) -> str:
    """Normalise countries for matching purposes, provided exclusively
    in addition to their raw counterparts."""

    country = country.upper()
    country = re.sub(r"[^\w\s]", " ", country)
    country = re.sub(r"\s+", " ", country)

    return country.strip()


@lru_cache
def strip_leading_articles(name):
    """Used to match normalised countries to their ISO codes."""
    if pd.isnull(name):
        return np.nan
    return re.sub(r"^THE\s+", "", name)


@lru_cache
def extract_modern_country(text):
    """Used to extract present-day countries from historical countries,
    to match countries to their ISO codes."""

    if text is None:
        return None

    text = text.upper()

    patterns = [
        r"\(NOW\s+([A-Z\s]+)\)",
        r"NOW\s+([A-Z\s]+)",
        r"CURRENTLY\s+([A-Z\s]+)"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(1).strip()

    return None


HISTORICAL_MAPPING = {
    "KAZAKH SSR": "KAZAKHSTAN",
    "KAZAKH SOVIET SOCIALIST REPUBLIC": "KAZAKHSTAN",
    "UKRAINIAN SSR": "UKRAINE",
    "BELARUSIAN SSR": "BELARUS",
    "MOLDOVIAN SSR": "MOLDOVA",
    "REPUBLIC OF MOLDOVA": "MOLDOVA",
    "REPUBLIC OF MOLDOVIA": "MOLDOVA",
    "RUSSIAN SFSR": "RUSSIAN FEDERATION",
    "RUSSIAN SOVIET FEDERATIVE SOCIALIST REPUBLIC RSFSR": "RUSSIAN FEDERATION",
    "RUSSIA USSR": "RUSSIAN FEDERATION",
    "YAKUT AUTONOMOUS SSR": "RUSSIAN FEDERATION",
    "KOMI ASSR": "RUSSIAN FEDERATION",
}


COUNTRY_SYNONYMS = {
    "UK": "UNITED KINGDOM",
    "U K": "UNITED KINGDOM",
    "GREAT BRITAIN": "UNITED KINGDOM",
    "RUSSIA": "RUSSIAN FEDERATION",
    "DPRK": "KOREA DEMOCRATIC PEOPLE S REPUBLIC OF",
    "NORTH KOREA": "KOREA DEMOCRATIC PEOPLE S REPUBLIC OF",
    "SOUTH KOREA": "KOREA REPUBLIC OF",
    "IRAN": "IRAN ISLAMIC REPUBLIC OF",
    "SYRIA": "SYRIAN ARAB REPUBLIC",
    "VIETNAM": "VIET NAM",
    "CAPE VERDE": "CABO VERDE",
    "ST LUCIA": "SAINT LUCIA",
    "ST KITTS AND NEVIS": "SAINT KITTS AND NEVIS",
    "CONGO DEMOCRATIC REPUBLIC": "CONGO THE DEMOCRATIC REPUBLIC OF THE",
    "CONGO (DEMOCRATIC REPUBLIC)": "CONGO THE DEMOCRATIC REPUBLIC OF THE",
    "PALESTINIAN": "PALESTINE",
    "OCCUPIED PALESTINIAN TERRITORIES": "PALESTINE",
}


def build_iso_mapping():
    """Build ISO map to derive ISO codes from country data."""

    mapping = {}

    for country in pycountry.countries:

        names = set()

        names.add(country.name)

        if hasattr(country, "official_name"):
            names.add(country.official_name)

        if hasattr(country, "common_name"):
            names.add(country.common_name)

        for name in names:
            norm = normalise_country(name)
            mapping[norm] = country.alpha_2

    return mapping


ISO_MAPPING = build_iso_mapping()


ISO_OVERRIDES = {
    "PALESTINE": "PS",
    "CONGO THE DEMOCRATIC REPUBLIC OF THE": "CD",
    "GAMBIA": "GM",
    "SAINT LUCIA": "LC",
    "SAINT KITTS AND NEVIS": "KN",
    "TURKEY": "TR",
}


@lru_cache
def resolve_country(raw_value):
    """Used to parse countries with inconsistent labels. Normalises and
    strips leading articles from countries, then attempts to match the
    normalisations to various maps such that ISO codes can be retrieved
    (where possible)."""

    result = {
        "raw_country": raw_value,
        "normalised_country": None,
        "iso_code": None,
        "iso_available": False,
        "resolution_method": None
    }

    if pd.isnull(raw_value):
        result["resolution_method"] = "missing"
        return result

    # normalize
    name = normalise_country(raw_value)

    # remove leading "THE"
    name = strip_leading_articles(name)

    # extract explicit modern country
    modern = extract_modern_country(name)

    if modern:
        name = modern
        result["resolution_method"] = "extracted_modern"

    # historical mapping
    if name in HISTORICAL_MAPPING:
        name = HISTORICAL_MAPPING[name]
        result["resolution_method"] = "historical_mapping"

    # apply synonyms
    if name in COUNTRY_SYNONYMS:
        name = COUNTRY_SYNONYMS[name]
        result["resolution_method"] = "synonym"
    else:
        result["resolution_method"] = "direct"

    result["normalised_country"] = name

    # ISO lookup
    if name in ISO_MAPPING:
        result["iso_code"] = ISO_MAPPING[name]
        result["iso_available"] = True
        return result

    # fallback overrides (official only)
    if name in ISO_OVERRIDES:
        result["iso_code"] = ISO_OVERRIDES[name]
        result["iso_available"] = True
        result["resolution_method"] = "override"
        return result

    # no ISO available
    result["resolution_method"] = "unmapped"
    return result


if __name__ == "__main__":

    # =======================================================================
    # LOAD RAW DATA
    # =======================================================================

    # load raw data as pandas dataframe
    df_raw = pd.read_csv('UK-Sanctions-List.csv', engine="python", dtype=str)

    # restructure daw data such that it is usable
    df = restructure_raw(df_raw)

    # =======================================================================
    # RENAME COLUMNS
    # =======================================================================
    # Normalise column names right after loading such that the column names
    # are consistent throughout the script.

    # normalise column names
    df.columns = [normalise_label(c) for c in df.columns]

    # handle awkward column names
    COLUMN_NAME_MAP = {
        # pluralisation
        "nationality_ies": "nationality",
        "business_registration_number_s": "business_registration_numbers",
        "current_owner_operator_s": "current_owners_operators",
        "previous_owner_operator_s": "previous_owners_operators",
        # clarification
        "designation_type": "subject_type",
    }
    df.columns = [COLUMN_NAME_MAP.get(c, c) for c in df.columns]

    # =======================================================================
    # CLEANING
    # =======================================================================

    # general cleaning pass removing leading/trailing whitespace
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    # replace empty strings with nans
    df = df.replace("", np.nan)

    # normalise categorical data -> lowercase_underscore_spaced_labels
    CATEGORICAL_COLUMNS_VALUES = {
        "name_type": {"primary_name", "alias", "primary_name_variation"},
        "subject_type": {"entity", "individual", "ship"},
        "alias_strength": {"good_quality_aka", "low_quality_aka"},
        "designation_source": {"un", "uk", "ukun"},
        "gender": {"male", "female"},
    }
    df = normalise_categorical_data(df, CATEGORICAL_COLUMNS_VALUES)

    # eager drop of duplicate rows
    df_dedup = df.drop_duplicates().reset_index()
    
    print(
        f"[INFO] Dropped {len(df) - len(df_dedup)} duplicate rows "
        f"({len(df)} -> {len(df_dedup)})"
    )
    
    df = df_dedup.copy()

    # =======================================================================
    # STRUCTURAL NORMALISATION OF NAMES
    # =======================================================================

    # define some useful masks
    INDIVIDUAL_MASK = df["subject_type"] == "individual"
    ENTITY_MASK = df["subject_type"] == "entity"
    SHIP_MASK = df["subject_type"] == "ship"

    # found anomalous rows (non-individual(s) with data under name_1-5)
    anomalous_rows = []

    for name_col in LATIN_NAME_COLS[:-1]:
        # criterion for 'anomalous' rows
        anomalous_mask = (ENTITY_MASK | SHIP_MASK) & (df[name_col].notna())
        # dataframe containing anomalous rows
        anomalies = df[anomalous_mask]
        # skip if no anomalous rows under this name column
        if len(anomalies) == 0:
            continue

        cols_to_print = ["unique_id", "subject_type", name_col, "name_6"]
        print(
            f"\n[WARNING] Detected rows with unexpected data in {name_col}:"
            f"\n{anomalies[cols_to_print].head()}"
        )
        
        # migrate the anomalous name data to the (assumedly) correct column
        # in the new rows
        anomalies["name_6"] = anomalies[name_col]
        # remove anomalous data from the rows that already exist in `df`
        df.loc[anomalous_mask, name_col] = [
            np.nan for _ in range(len(anomalies))
        ]
        # add to list for later concatenation with `df`    
        anomalous_rows.append(anomalies.drop_duplicates())

    # concatenate with `df`
    if len(anomalous_rows) > 0:
        for anomalies in anomalous_rows:
            df = pd.concat([df, anomalies], axis=0, ignore_index=True)

    # =======================================================================
    # CONSTRUCTION
    # =======================================================================

    # add row_id
    get_id = lambda x, L: f"{x:0>{len(str(L - 1))}}" 
    df["id"] = [get_id(i, len(df)) for i in range(len(df))]

    # build tables:
    # - SUBJECTS
    # - SUBJECT_NAMES
    # - SUBJECT_DOBS
    # - SUBJECT_COUNTRIES
    # - SUBJECT_IDENTIFIERS

    # -----------------------------------------------------------------------
    # SUBJECTS
    # -----------------------------------------------------------------------

    COLUMNS_SUBJECTS = [
        "unique_id",
        "subject_type",
        "un_reference_number",
        "ofsi_group_id",
        "date_designated",
        "last_updated",
    ]

    subjects = df[COLUMNS_SUBJECTS].drop_duplicates()

    # -----------------------------------------------------------------------
    # SUBJECT_NAMES
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("SUBJECT_NAMES")
    print("-" * 80)

    COLUMNS_SUBJECT_NAMES = [
        "id",
        "unique_id",
        "name_type",
        "name_1",
        "name_2",
        "name_3",
        "name_4",
        "name_5",
        "name_6",
        "alias_strength",
    ]

    # select rows with name data
    name_mask = (
        (df[LATIN_NAME_COLS[0]].notna()) |
        (df[LATIN_NAME_COLS[1]].notna()) |
        (df[LATIN_NAME_COLS[2]].notna()) |
        (df[LATIN_NAME_COLS[3]].notna()) |
        (df[LATIN_NAME_COLS[4]].notna()) |
        (df[LATIN_NAME_COLS[5]].notna())
    )
    subject_names = df.loc[name_mask, COLUMNS_SUBJECT_NAMES]

    print(
        f"[INFO] Initial SUBJECT_NAMES:"
        f"\n{subject_names.columns.tolist()}"
        f"\nrows: {len(subject_names):,}"
        f"\ncols: {len(subject_names.columns):,}"
    )

    # rename name_type values (drop excessive use of "name")
    subject_names["name_type"] = (
        subject_names["name_type"]
        .replace({
            "primary_name": "primary",
            "primary_name_variation": "primary_variation",
        })
    )

    # build full_name column
    subject_names[FULL_NAME_COL] = (
        subject_names[LATIN_NAME_COLS]
        .apply(build_full_name, axis=1)
        .replace("", np.nan)
    )

    # drop exact duplicate names
    n_names_pre_dedup = len(subject_names)

    subject_names = (
        subject_names
        .drop_duplicates(["unique_id", "name_type", *LATIN_NAME_COLS])
    )
    
    n_names_dedup = len(subject_names)
    
    if n_names_pre_dedup > n_names_dedup:
        print(
            f"[INFO] Removed {n_names_pre_dedup - n_names_dedup:,} "
            f"rows with duplicate names "
            f"-> {n_names_dedup:,} rows"
        )

    # remove least complete names
    subject_names = remove_least_complete_names(subject_names)

    n_names_most_complete = len(subject_names)
    
    if n_names_dedup > n_names_most_complete:
        print(
            f"[INFO] Removed {n_names_dedup - n_names_most_complete:,} "
            f"rows with redundant names "
            f"-> {n_names_most_complete:,} rows"
        )

    # add name_id
    subject_names["name_id"] = [
        get_id(i, len(subject_names))
        for i in range(len(subject_names))
    ]

    # sort by unique_id, name_type
    subject_names["priority"] = (
        subject_names["name_type"]
        .replace({"primary": 3, "primary_variation": 2, "alias": 1})
    )
    
    subject_names = (
        subject_names
        .sort_values(by=["unique_id", "priority"], ascending=[True, False])
        .drop(columns=["priority"])
    )

    # reorder columns
    subject_names_cols = [
        "name_id",
        "unique_id",
        "name_type",
        "alias_strength",
        "full_name",
        "name_6",
        "name_1",
        "name_2",
        "name_3",
        "name_4",
        "name_5",
    ]

    subject_names = subject_names[subject_names_cols]

    print(
        f"[INFO] Final SUBJECT_NAMES:"
        f"\n{subject_names.columns.tolist()}"
        f"\nrows: {len(subject_names):,}"
        f"\ncols: {len(subject_names.columns):,}"
    )

    # NOTE: duplicate and redundant names have been removed
    #       within each unique_id, but this could be further
    #       reduced to fully unique names
    #       -> name-to-unique_id could be many-to-one
    #       -> would require mapping name to unique_id list

    # -----------------------------------------------------------------------
    # SUBJECT_DOBS
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("SUBJECT_DOBS")
    print("-" * 80)

    COLUMNS_SUBJECT_DOBS = [
        "id",
        "unique_id",
        "dob",
    ]

    subject_dobs = df.loc[df["dob"].notna(), COLUMNS_SUBJECT_DOBS]

    print(
        f"[INFO] Initial SUBJECT_DOBS:"
        f"\n{subject_dobs.columns.tolist()}"
        f"\nrows: {len(subject_dobs):,}"
        f"\ncols: {len(subject_dobs.columns):,}"
    )

    subject_dobs_parsed = subject_dobs["dob"].apply(parse_dob)
    subject_dobs_parsed_df = pd.DataFrame(subject_dobs_parsed.tolist())
    subject_dobs = pd.concat(
        [
            subject_dobs,
            subject_dobs_parsed_df.drop(columns=["raw_dob"])
        ],
        axis=1
    )

    # rename columns (remove excessive use of "dob")
    subject_dobs.columns = [
        # canonical columns (`COLUMNS_SUBJECT_DOBS`)
        "id",
        "unique_id",
        "raw_value",
        # extra columns added from D.O.B parsing
        "year",
        "month",
        "day",
        "precision",
        "is_valid",
        "parse_issues",
    ]

    # check all D.O.Bs included are valid
    n_invalid = len(subject_dobs) - subject_dobs["is_valid"].sum()
    if n_invalid > 0:
        print(f"[WARNING] {n_invalid:,} invalid D.O.Bs detected")

        # remove rows with invalid D.O.Bs
        dob_mask = subject_dobs["is_valid"] == True
        subject_dobs = subject_dobs[dob_mask]

        print(f"[INFO] Removed {len(dob_mask) - len(subject_dobs):,} invalid D.O.Bs")

    print(f"[INFO] All {len(subject_dobs):,} D.O.B rows have valid D.O.Bs")

    # join lists of strings in parse_issues
    subject_dobs["parse_issues"] = (
        subject_dobs["parse_issues"]
        .apply(", ".join)
        .replace("", np.nan)
    )

    # drop is_valid column (all D.O.Bs are now valid)
    subject_dobs = subject_dobs.drop(columns=["is_valid"])

    # drop exact duplicate D.O.Bs
    n_dobs_pre_dedup = len(subject_dobs)
    
    subject_dobs = (
        subject_dobs
        .drop_duplicates(subset=["unique_id", "year", "month", "day"])
    )

    n_dobs_dedup = len(subject_dobs)
    
    if n_dobs_pre_dedup > n_dobs_dedup:
        print(
            f"[INFO] Removed {n_dobs_pre_dedup - n_dobs_dedup:,} "
            f"rows with duplicate D.O.Bs "
            f"-> {n_dobs_dedup:,} rows"
        )

    # remove least complete D.O.Bs
    subject_dobs = remove_least_complete_dobs(subject_dobs)
    
    n_dobs_most_complete = len(subject_dobs)
    
    if n_dobs_dedup > n_dobs_most_complete:
        print(
            f"[INFO] Removed {n_dobs_dedup - n_dobs_most_complete:,} "
            f"rows with redundant D.O.Bs "
            f"-> {n_dobs_most_complete:,} rows"
        )

    # add dob_id
    subject_dobs["dob_id"] = [
        get_id(i, len(subject_dobs))
        for i in range(len(subject_dobs))
    ]

    # sort by unique_id, precision
    subject_dobs["priority"] = (
        subject_dobs[["year", "month", "day"]]
        .notna()
        .sum(axis=1)
    )

    subject_dobs = (
        subject_dobs
        .sort_values(by=["unique_id", "priority"], ascending=[True, False])
        .drop(columns=["priority"])
    )

    # reorder columns
    subject_dobs_cols = [
        "dob_id",
        "unique_id",
        "year",
        "month",
        "day",
        "precision",
        "parse_issues",
    ]

    subject_dobs = subject_dobs[subject_dobs_cols]

    print(
        f"[INFO] Final SUBJECT_DOBS:"
        f"\n{subject_dobs.columns.tolist()}"
        f"\nrows: {len(subject_dobs):,}"
        f"\ncols: {len(subject_dobs.columns):,}"
    )

    # NOTE: duplicate and redundant D.O.Bs have been removed
    #       within each unique_id, but this could be further
    #       reduced to fully unique D.O.Bs
    #       -> D.O.B-to-unique_id could be many-to-one
    #       -> would require mapping D.O.B to unique_id list

    # -----------------------------------------------------------------------
    # SUBJECT_COUNTRIES
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("SUBJECT_COUNTRIES")
    print("-" * 80)

    # columns for dataframe initialisation
    COLUMNS_SUBJECT_COUNTRIES = [
        "id",
        "unique_id",
    ]

    # construction list
    subject_countries_types: list[pd.DataFrame] = []

    # different types of identifiers to add to the dataframe
    subject_country_types = [
        "country_of_birth",
        "address_country",
        "nationality"
    ]

    # populate construction list
    for c in subject_country_types:

        type_mask = df[c].notna()  # select rows with identifier data
        cols = [*COLUMNS_SUBJECT_COUNTRIES, c]
        # filter for desired rows and columns
        subject_countries_type = df.loc[type_mask, cols]
        # rename `c` to raw_value and migrate `c` to country_type
        subject_countries_type.columns = [*COLUMNS_SUBJECT_COUNTRIES, "raw_value"]
        subject_countries_type["country_type"] = c
        # add dataframe to construction list
        subject_countries_types.append(subject_countries_type)
    
    # construct dataframe
    subject_countries = pd.concat(subject_countries_types, axis=0)

    print(
        f"[INFO] Initial SUBJECT_COUNTRIES:"
        f"\n{subject_countries.columns.tolist()}"
        f"\nrows: {len(subject_countries):,}"
        f"\ncols: {len(subject_countries.columns):,}"
    )

    # rename country_type values
    subject_countries["country_type"] = (
        subject_countries["country_type"]
        .replace({
            "country_of_birth": "birth",
            "address_country": "address",
        })
    )

    # parse countries
    subject_countries["parsed"] = (
        subject_countries["raw_value"]
        .apply(resolve_country)
    )

    # add iso codes
    subject_countries["iso"] = (
        subject_countries["parsed"]
        .apply(lambda x: x["iso_code"] if x["iso_available"] else np.nan)
    )

    # add method for country resolution
    subject_countries["resolution_method"] = (
        subject_countries["parsed"]
        .apply(lambda x: x["resolution_method"])
    )

    # add iso_available flag
    subject_countries["iso_available"] = (
        subject_countries["parsed"]
        .apply(lambda x: x["iso_available"])
    )

    # drop parsed column (desired data already extracted)
    subject_countries = (
        subject_countries
        .drop(columns=["parsed"])
    )

    print(
        f"[INFO] Final SUBJECT_COUNTRIES:"
        f"\n{subject_countries.columns.tolist()}"
        f"\nrows: {len(subject_countries):,}"
        f"\ncols: {len(subject_countries.columns):,}"
    )

    # -----------------------------------------------------------------------
    # SUBJECT_IDENTIFIERS
    # -----------------------------------------------------------------------

    print("-" * 80)
    print("SUBJECT_IDENTIFIERS")
    print("-" * 80)

    # columns for dataframe initialisation
    COLUMNS_SUBJECT_IDENTIFIERS = [
        "id",
        "unique_id",
        "ofsi_group_id",
    ]

    # construction list
    subject_identifier_types: list[pd.DataFrame] = []

    # different types of identifiers to add to the dataframe
    id_cols = [
        "national_identifier_number",
        "passport_number",
        "business_registration_numbers",
        "imo_number",
        "hull_identification_number_hin"
    ]

    # populate construction list
    for c in id_cols:

        type_mask = df[c].notna()  # select rows with identifier data
        cols = [*COLUMNS_SUBJECT_IDENTIFIERS, c]
        # filter for desired rows and columns
        subject_identifier_type = df.loc[type_mask, cols]
        # rename `c` to raw_value and migrate `c` to identifier_type
        subject_identifier_type.columns = [*COLUMNS_SUBJECT_IDENTIFIERS, "raw_value"]
        subject_identifier_type["identifier_type"] = c
        # add dataframe to construction list
        subject_identifier_types.append(subject_identifier_type)
    
    # construct dataframe
    subject_identifiers = pd.concat(subject_identifier_types, axis=0)

    # rename identifier_type values
    subject_identifiers["identifier_type"] = (
        subject_identifiers["identifier_type"].replace({
            "national_identifier_number": "national",
            "passport_number": "passport",
            "imo_number": "imo",
            "hull_identification_number_hin": "hin"
        })
    )

    # get additional_info for passport data
    passport_mask = subject_identifiers["identifier_type"] == "passport"
    subject_identifiers.loc[passport_mask, "additional_info"] = (
        df.loc[
            df["passport_number"].notna(),
            "passport_additional_information"
        ]
    )

    # get additional_info for national id data
    nationalid_mask = subject_identifiers["identifier_type"] == "national"
    subject_identifiers.loc[nationalid_mask, "additional_info"] = (
        df.loc[
            df["national_identifier_number"].notna(),
            "national_identifier_additional_information"
        ]
    )

    print(
        f"[INFO] SUBJECT_IDENTIFIERS:"
        f"\n{subject_identifiers.columns.tolist()}"
        f"\nrows: {len(subject_identifiers):,}"
        f"\ncols: {len(subject_identifiers.columns):,}"
    )

    # =======================================================================
    # EXPORT
    # =======================================================================

    print("-" * 80)
    print("EXPORT")
    print("-" * 80)

    OUTPUT_PATH = "uk_sanctions_transformed.xlsx"

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        # df.to_excel(writer, sheet_name="raw", index=False)
        subjects.to_excel(writer, sheet_name="subjects", index=False)
        subject_names.to_excel(writer, sheet_name="subject_names", index=False)
        subject_countries.to_excel(writer, sheet_name="subject_countries", index=False)
        subject_identifiers.to_excel(writer, sheet_name="subject_identifiers", index=False)

    print(f"[INFO] Output written to {OUTPUT_PATH}")

    CSV_FILES = {
        # "raw.csv": df,
        "subjects.csv": subjects,
        "subject_names.csv": subject_names,
        "subject_dobs.csv": subject_dobs,
        "subject_countries.csv": subject_countries,
        "subject_identifiers.csv": subject_identifiers,
    }

    for filename, table in CSV_FILES.items():
        table.to_csv(filename, index=False)
        print(f"[INFO] CSV written to {filename}")
