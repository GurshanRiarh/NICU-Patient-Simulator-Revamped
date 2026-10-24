# PreMo

Deidentified export of vital signs and REDCap clinical data for the PreMo 4-NICU study of  very low birth weight infants

## Contents

- redcap: Folder for REDCap form exports
  - 'redcap.Demographic and Perinatal data.parquet': Demographic data
  - 'redcap.Discharge & Outcomes data.parquet'
  - 'redcap.Intubation eval data.parquet'
  - 'redcap.Sepsis or NEC eval data.parquet'
  - redcap.metadata.parquet: RedCAP metadata describing columns
  - redcap.premo_deid.xlsx: parquet files exported as MS Excel workbook
- vitals: Folder for 0.2hz - 1hz heart rate and SP02 vital signs for 3829 patients
  - XXXXXXX.parquet, where XXXXXXX indicates the subject_number_deid deidentified REDCap patient identifier
- scripts: Folder for example scripts plotting the data, generating the gtsummary table, and generating the ro-crate
  - condense_crate.py: condense the generated ro-crate
  - generate_redcap_metadata.py: generate ro-crate
  - gtsummary_table.R: generate gtsummary table (without vitals coverage)
  - plot_patient_segment.m: plot patient segment in matlab
  - plot_patient_segment.py: plot patient segment in python using neurokit2 and matplotlib
- ro-crate-metadata.json: ro-crate metadata
- ro-crate-preview.html: generated ro-crate webpage
- ro-crate-prov-graph.html: generated ro-crate provenance graph webpage
- ro-crate-prov-graph.json: generated ro-crate provenance graph json data

## Fairscape

This dataset is loaded into Fairscape at https://fairscape.net

## Deidentification notes

- All dates have been converted to ages and stored as parquet durations (for redcap files) or integer based timestamps (for vitals files).
- REDCap subject_numbers have been deidentified to subject_numbers_deid
- Site names were removed.
