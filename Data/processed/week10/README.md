# Week 10 processed data

The normal full-data run writes only first-500-row **preview CSV** files for training, validation and test. All metrics and model fitting nevertheless use the **full eligible partitions**. The optional `--save-full-data` command writes large complete `.csv.gz` partitions and is not needed for the assignment; keep them out of Git unless authorized and size-appropriate.
