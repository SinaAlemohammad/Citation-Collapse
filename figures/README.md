# Reproduction package: The AI Citation Trap

Reproduces every statistic and every matplotlib figure in the paper from the
four canonical CSVs. One command:

    python reproduce_all.py            # all stats + all figures (~5 min)
    python reproduce_all.py --stats    # stats only

## Inputs (place in this folder)

    node0-all-models.csv                 round-0 selections, 11 models x 120 prompts
    all-models-all-nodes-selections.csv  recursion, 8 models x 12 rounds x 120 prompts
    node0_human_responses.csv            expert annotations (8 annotators, 53 prompts)
    node0_seed_categories.csv            the 120 seed papers (titles + abstracts)

## Outputs

    stats_report.txt   every headline number, in paper order
    *.json, *.npy      intermediate caches (the schema the figure scripts read)
    figures/*.pdf      F1 teaser; F3 heatmap + votes; F4 map identity;
                       F5 symmetry; F6 recursion; F7 mechanism;
                       FD1, FE1, FF1, FI1 (appendices)

Figure F2 (the pipeline diagram) is TikZ drawn inside the LaTeX source, not a
matplotlib figure, so it is not produced here.

## Module map

    repro_core.py       loading, matched-null redraws, concentration metrics
    repro_maps.py       maps, split-half reliability, disattenuated agreement,
                        PC1, vendor blocks, plurality floor, exchangeability,
                        human populations (App. E, all three)
    repro_extra.py      rho / stable rank (Prop. 4), HHI identity (App. D),
                        vote histogram (App. C.3), text model (App. F)
    repro_recursion.py  per-round recursion metrics (Sec. 4, App. I),
                        within-panel competition test + placebo (App. G)
    reproduce_all.py    orchestrator; writes caches, then runs the four
                        figure scripts unchanged
    make_*.py           the exact figure scripts used for the paper

## Reproduction tolerances (verified against the paper's pipeline values)

Deterministic quantities reproduce exactly: per-model top-10% shares, HHI,
exclusion counts, hallucination rates, PC1 (73.3%), all six vendor blocks,
human top-10%/exclusions for all three populations, chi-square statistics
(130.7 / 125.9 / 125.8), regression row and cluster counts (7,557 / 1,292),
beta_gen (all eight, exact).

Monte Carlo quantities reproduce to redraw tolerance (~0.2 pp): matched
nulls (NREDRAW=150 at round 0, NRED_R=60 in the recursion), the plurality
floor (72-73%), and the recursion null trajectories (e.g. 29.1% / 38.0%
round-11 values). The within-panel betas use alternating demeaning for the
two-way fixed effects and match the pipeline's dummy-variable implementation
to within 0.05 pp, the tolerance disclosed in App. G. rho and stable rank
match to a few hundredths. The exclusion series is reported as a fraction of
shown papers; churn is stored as top-decile retention (the paper quotes
1 - retention).

## Requirements

    numpy  pandas  scipy  scikit-learn  matplotlib

(see requirements.txt; no network, no API keys, no GPU)
