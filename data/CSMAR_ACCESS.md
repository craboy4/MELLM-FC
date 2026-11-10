# Obtaining CSMAR data and preparing the research panel

This guide covers authorized access, required exports, local preparation, and reproduction. Real CSMAR rows and case narratives are **not included**. A subscription provides access under its terms, not automatic permission to redistribute an exported sample.

## 1. Obtain authorized access

Start at the [CSMAR data portal](https://data.csmar.com/) or your institution's library database page. Confirm that your subscription covers the required modules. Without institutional access, contact CSMAR about an appropriate research license.

Follow your library's instructions for personal registration, institutional authentication, and permitted off-campus access. A personal account does not necessarily unlock every module. CSMAR describes institutional association in its [account-access guide](https://www.csmar.com/contents/9/861.html); current screens may differ from that 2022 guide. Campus-network or library-VPN requirements depend on your institution. See this [university library access guide](https://lib.zufe.edu.cn/databaseguide/detail/45) for an institutional example.

## 2. Locate the required data

Search tables and fields and obtain their data dictionaries. These are research concepts, **not verified vendor table IDs or field codes**; check the current dictionary before choosing fields.

| Data category | Required information | Purpose |
| --- | --- | --- |
| Listed-company information and industry classification | Stable identifier, reporting year, industry | Firm-year matching and exclusion of financial firms |
| Financial statements and notes | Depreciation/amortization; assets, liabilities, capital; revenue, costs, expenses, profit | N1-N17 and N20-N25 |
| Share capital and stock-market data | Tradable shares; closing price on the last trading day of each year | N18 and N19 |
| Regulatory penalties and exchange violation records | Event identifier, company, violation category, event years, announcement dates | Financial-disclosure violation labels |
| Original violation narratives, if licensed | Activity text and source/provenance fields | Optional development-stage case feedback |

Fixed-formula reproduction needs the numerical panel and labels, **not** narratives or an LLM API. Follow [variable_dictionary.csv](variable_dictionary.csv) for all 25 definitions. N9 is taxes payable; N23 is income tax expense; N25 is revenue minus operating cost. N15 and N16 are the specified component sums. Do not substitute similarly named ratios or confuse expenses with cash payments.

## 3. Export and record the extraction locally

Query annual reporting years **2010-2024** for Chinese non-financial listed companies. Exclude financial firms using the CSMAR industry classification, as specified in the manuscript. Export through your authorized interface within institutional download rules. The [library query guide](https://lib.zufe.edu.cn/databaseguide/detail/45) describes date, code, field, and filter selection and CSV/spreadsheet export options. No downloader or authentication bypass is supplied here.

Keep an extraction log and original exports in a private directory. Record:

- Table/field identifiers, dictionary version, extraction date, query dates, universe, industry scheme, exclusion codes, and every additional filter.
- Annual-statement type, consolidated/parent-company basis, original/restated version, and duplicate-statement resolution policy.
- Monetary, share-count, and price units; price-adjustment conventions; share-count observation date.
- Join keys, component definitions, label-category mapping, and SHA-256 checksums of original exports.

**Known reconstruction gap:** this release lacks a complete original vendor query log, field-code mapping, statement-version policy, and every row-level inclusion decision. Do not infer those settings from final row counts or treat the examples above as undocumented original settings. Resolve missing choices from original preparation records through the authors and your institution where permitted. Otherwise, document the differences and describe an independently reconstructed sample. Later database revisions and enforcement disclosures may change a fresh export. This guide does not guarantee exact reconstruction.

## 4. Build labels and numerical inputs

Join by stable company identifier and reporting year. Resolve duplicate statements using a documented policy; do not select arbitrary duplicates or mix annual and interim statements. Preserve annual-record coverage. Historical operators use available records, so inserting synthetic missing firm-years changes the calculation.

The manuscript labels **recorded financial-disclosure violations**, broader than proven intentional material misstatement. Positive categories include inflated profit, misstated assets, false records, misleading statements, material omission, and inaccurate disclosure. Map vendor codes using the relevant dictionary and original labeling records; not every regulatory violation qualifies.

Assign events to their recorded violation year(s), not announcement years: an event in year t announced in t+2 belongs to t. Expand multi-year events only through documented associations. Label 0 means no qualifying recorded violation under this rule, not proof of no fraud. Retain an event-to-firm-year linkage table locally. These retrospective labels do not establish point-in-time availability.

Prepare a UTF-8 CSV containing exactly:

```text
firm_id,year,label,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25
```

- `firm_id`: stable string grouping key, not a predictor. Use zero-padded codes or order-preserving surrogate IDs; firm ordering affects tied ranking metrics.
- `year`: integer from 2010 through 2024, with one row per firm-year.
- `label`: binary 0/1 under the rule above.
- `N1`-`N25`: original quantities and units. Retain float64 precision with 17 significant digits. Use empty fields for missing values, not zeros or synthetic replacements.

Save as `private_data/panel.csv`, a Git-ignored location. Do not add full-sample imputation, standardization, winsorization, manual ratios, random splitting, or rebalanced sampling. The runner fits base-variable medians on the appropriate training subset.

## 5. Validate before fitting

After installing dependencies from the [main README](../README.md), run:

```sh
python validate_data.py --data private_data/panel.csv --out outputs/data_check.json
```

This offline check reports schema validity and aggregate counts. It does not fit models, contact an API, publish records, or repair labels. Matching counts is **necessary but insufficient** for sample identity.

| Sample | Years | Rows |
| --- | --- | ---: |
| Development training | 2010-2018 | 25,294 |
| Development validation | 2019-2021 | 13,072 |
| Final training | 2010-2021 | 38,366 |
| Final test | 2022-2024 | 15,459 |
| Entire panel | 2010-2024 | 53,825 |

The original panel has 5,400 firms and 6,556 positives overall; the test period has 1,670 positives. Investigate mismatches in exports, duplicates, universe definitions, event-year associations, or database revisions. Do not force counts to match by adding/dropping arbitrary rows, or tune the model to match reference test scores.

## 6. Reproduce the fixed numerical result

```sh
python reproduce.py --data private_data/panel.csv --out outputs/main
python compare_results.py outputs/main/metrics.json
```

Use a new output directory for each independent run. The runner computes all 60 formulas and refits Base and full models using both prescribed training periods. Reference scores never enter fitting. Formula order, units, precision, cohort, labels, software, and recipe all matter. Local verification used the original licensed panel, not an independently downloaded reconstruction. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for numerical scope.

## 7. Optional case-based generation

Only new case-based generation needs narratives. Follow the [development-case JSON contract](README.md#development-cases). Preserve genuine text and source references; do not invent cases or substitute another dataset's cases. Event-year associations must be confined to 2010-2021 and must not introduce test-sample facts. Later publication dates can occur in these retrospective records; this does not establish contemporaneous information availability.

The case matcher requires numeric security codes encoded consistently in the panel and case JSON. Surrogate IDs acceptable for fixed-formula reproduction are not automatically suitable for this matcher. Keep cases, requests, and responses private and use an endpoint authorized to receive development context. Public source, prompts, and comments are English; locally supplied original-language narratives are not translated by this release.

## Attribution and sharing

Cite CSMAR and document modules, extraction date, and preprocessing. Provider and institutional terms still apply. The repository's noncommercial research license covers the authors' contributions, not CSMAR data. Do not attach licensed exports, narratives, account details, or row-level predictions to GitHub issues. Attribution, a noncommercial-use notice, or a takedown promise does not replace redistribution permission.
