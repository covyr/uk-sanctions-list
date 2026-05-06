## UK Sanctions List

Code assignment project. Transform a UK sanctions list such that it can be used for comparison against customer records.

# Setup

Install the required Python packages:

    python3 -m pip install -r requirements.txt

Run the script:

    python -m transform.py

## Data

# Column Names

Clean and normalise column names (strip whitespace, force lowercase, replace spaces/slashes with underscores, remove punctuation except underscores, collapse repeated underscores)
    e.g. `Name type` -> `name_type`
Further rename some columns to whom name normalisation is not kind
    e.g. `nationality_ies` -> `nationality`

This is performed immediately after loading the data for consistency purposes.

# Duplicate Data

Removed 651 rows of duplicate data.

# Data Inconsistency

Columns containing categorical data, whose meaning is not changed by normalisation, are normalised early in the script so that categorical data is accessed via labels that are consistent throughout the script.

The `name_1`, `name_2`, `name_3`, `name_4`, and `name_5` columns should only be used for `individual` subject types, but there are instances within the data of `entity` subject types with data under these columns. To account for this, the values under these columns are used to create new rows, with these values migrated to `name_6`, where they carry the same significance as the already present data under `name_6`, such that all `entity` subject type name parts are under `name_6`.

Names are parsed and normalised into `full_name` and `compact_name` forms for matching purposes.

D.O.B values demonstrated variability in terms of their format and completeness. `dd` and `mm` are consistently used as placeholders for unknown days and months, respectively. There are also D.O.B values with only year data, as well as one particular value with `00/00` representing `dd/mm`, which was treated as an unknown value. D.O.Bs are parsed and separated into year, month, and day parts (where possible), also flagging precision and validity, to acknowledge incomplete data.

Additional information on `passport_number` and `national_identifier_number` seems very metadata-rich, would be good data to parse. 

NOTE: Normalised values/representations are exclusively provided in addition to their raw, untouched counterparts. These normalisations (names especially) if used as fully true values have high potential to introduce elevated risk of false-positive matches.