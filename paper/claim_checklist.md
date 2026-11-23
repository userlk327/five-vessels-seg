# S1 Checklist. CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging, 2024 update)

Items and titles from Tejani et al., Radiology: Artificial Intelligence 2024;6(4):e240300 (doi:10.1148/ryai.240300),
taken from the full text (PMC11304031) on 2026-11-23; locations updated after the Step 4 revision. Location: section of the manuscript (S1 Appendix = supplementary
methods, tables and figures). Answer: Yes / Partly / No / NA, with a short note where needed.

| # | Item | Answer | Location and note |
|---|---|---|---|
| 1 | Identification as a study of AI methodology, specifying the category of technology used | Yes | Title, Abstract (convolutional networks, Segment Anything Model, low-rank adaptation) |
| 2 | Summary of study design, methods, results, and conclusions | Yes | Abstract |
| 3 | Scientific and/or clinical background, including the intended use and role of the AI approach | Yes | Introduction (methodological comparison; no clinical deployment intended) |
| 4 | Study aims, objectives, and hypotheses | Yes | Introduction (three research questions) |
| 5 | Prospective or retrospective study | Yes | Methods, Data, audit and split (retrospective analysis of a public dataset) |
| 6 | Study goal | Yes | Introduction; Methods, Study design (controlled comparison of model configurations) |
| 7 | Data sources | Yes | Methods, Data, audit and split; Declarations (FIVES; Hugging Face copy) |
| 8 | Inclusion and exclusion criteria | Yes | Methods, Data, audit and split; Table 1 (three exclusions with reasons) |
| 9 | Data preprocessing | Yes | Methods, Preprocessing and training targets; S1 Appendix A3 |
| 10 | Selection of data subsets | Yes | Methods, Data, audit and split (grouped split); S1 Appendix A1 |
| 11 | De-identification | Partly | Ethics statement: the FIVES paper does not describe de-identification, and some images carry burned-in text that may contain name fragments; the markings were never read, figures pixelate the area outside the FOV, and no image data, marking locations or crops are released |
| 12 | Missing data | Yes | Methods and Table 1 (two empty masks excluded; no other missing data) |
| 13 | Image acquisition protocol | Yes | Methods, Data (Second Affiliated Hospital of Zhejiang University, 2016–2021, Topcon TRC-NW8, 50° field of view, 2048 × 2048 colour fundus photographs) |
| 14 | Definition of method(s) used to obtain the reference standard | Yes | Methods, Data (pixelwise manual annotation provided with FIVES) |
| 15 | Rationale for choosing the reference standard | Yes | Methods, Data (the dataset's own annotations; masks not edited) |
| 16 | Source of reference standard annotations | Yes | Methods, Data (trained medical staff, verified by ophthalmologists, per the dataset paper) |
| 17 | Annotation of test set | Yes | Methods, Data (official test masks used as provided) |
| 18 | Measures of inter- and intrarater variability | Partly | Methods and Results, Agreement between two available annotations (post hoc, n = 6; annotators unknown) |
| 19 | How data were assigned to partitions | Yes | Methods, Data, audit and split; Table 1; S1 Appendix A1 |
| 20 | Level at which partitions are disjoint | Yes | Methods (image level with grouping of likely same-patient images); Discussion, Limitations (patient-level separation not guaranteed) |
| 21 | Intended sample size | Yes | Methods, Data (sample size fixed by the dataset; no sample-size calculation) |
| 22 | Detailed description of model | Yes | Methods, Models; Table 2; S1 Appendix A5; public code |
| 23 | Software libraries, frameworks, and packages | Yes | Methods, Computational cost and software |
| 24 | Initialization of model parameters | Yes | Methods, Models and Training (CNNs from seeded random initialization; SAM from the released ViT-B weights; LoRA initialization in S1 Appendix A5) |
| 25 | Details of training approach | Yes | Methods, Training; S1 Appendix A4, A5, A8; S5 Table; S1 Fig (training curves) |
| 26 | Method of selecting the final model | Yes | Methods, Training and Evaluation (best validation Dice checkpoint; threshold chosen on validation) |
| 27 | Ensembling technique | NA | No ensembling; seeds are reported separately (Methods, Training) |
| 28 | Metrics of model performance | Yes | Methods, Evaluation |
| 29 | Statistical measures of significance and uncertainty | Yes | Methods, Statistical analysis; post hoc cluster and seed-and-cluster bootstraps (Methods, Analysis decisions and later additions; S1 Appendix A9; S16 Table) |
| 30 | Robustness or sensitivity analysis | Yes | Fixed-threshold analysis, threshold curves, alternative width cut-offs, three seeds, simple effects of the 2 × 2 design, seam check (Methods; Results; Fig 5; S2 Fig; S3, S8, S9, S10 Tables); post hoc: matched precision, width-stratified precision, spur-pruned thin class, clDice and NSD at 0.5, resampling of clusters and seeds (Results, Post hoc analyses; S14–S16 Tables; S7 Fig) |
| 31 | Methods for explainability or interpretability | No | Not studied; qualitative error maps only (Fig 7) |
| 32 | Evaluation on internal data | Yes | Results (official FIVES test set) |
| 33 | Testing on external data | No | Discussion, Limitations (no external testing) |
| 34 | Clinical trial registration | NA | Not a clinical trial |
| 35 | Numbers of patients or examinations included and excluded | Partly | Table 1 (images per partition and exclusions); patients per partition unknown (no patient identifiers) |
| 36 | Demographic and clinical characteristics of cases in each partition and dataset | Partly | Methods, Data (dataset-level age, sex and ethnicity from the dataset paper); disease category per partition in Table 1; demographics per partition not available |
| 37 | Performance metrics and measures of statistical uncertainty | Yes | Results; Tables 3–5; S6, S7, S14–S16 Tables |
| 38 | Estimates of diagnostic performance and their precision | NA | Segmentation study; no diagnostic task |
| 39 | Failure analysis of incorrect results | Yes | Results (qualitative error maps, Fig 7, S3 Fig; disease categories, S11 Table, S4 Fig; seam check, S3 Table; annotation agreement, S12 Table) |
| 40 | Study limitations | Yes | Discussion, Limitations |
| 41 | Implications for practice, including intended use and/or clinical role | Yes | Discussion, Comparison with published results and recommendations (for method comparison; no clinical use claimed) |
| 42 | Provide a reference to the full study protocol or to additional technical details | Yes | Declarations, Code availability (protocol document in the public repository); S1 Appendix |
| 43 | Statement about the availability of software, trained model, and/or data | Yes | Declarations, Data and Code availability [TODO: trained weights are not yet published] |
| 44 | Sources of funding and other support; role of funders | [TODO] | Declarations, Funding |
