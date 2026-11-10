# Input data

Obtain the licensed panel and case records through the [CSMAR guide](CSMAR_ACCESS.md). `synthetic_demo.csv` contains 30 fictional firms over 2010-2024. Field definitions appear in `variable_dictionary.csv`.

## Panel

CSV columns are `firm_id,year,label,N1,...,N25`, with one row per firm-year. Keep company identifiers consistent and preserve their ordering for tied ranks. Years are integers from 2010 to 2024 and labels are binary. Keep original units and float64 precision, with empty fields for missing values.

The paper panel has 53,825 rows and 5,400 firms. Development training and validation contain 25,294 and 13,072 rows. Final training and test contain 38,366 and 15,459 rows, including 1,670 test positives.

## Development cases

New generation accepts a JSON list with `Symbol`, `linked_years`, `ViolationID`, `Activity` and `source_record`. Keep original provenance and publication dates. Symbols must match numeric panel identifiers. Event-year associations must fall within 2010-2021, with later event facts excluded. Some source narratives were published later, so the evaluation is retrospective.

Keep licensed records and generated run outputs local. Data use follows provider and institutional terms.
