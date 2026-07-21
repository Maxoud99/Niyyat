# Quick Reference: Timestamped Outputs

## Run the Script
```bash
cd clustering-organized/scripts
python3 compare_clustering_algorithms.py
```

## Output Location
```
outputs/run_YYYYMMDD_HHMMSS/
├── results/
├── plots/
└── logs/
```

## Find Latest Run
```bash
ls -t outputs/run_* | head -1
```

## Find All Results
```bash
find outputs -name "algorithm_comparison.csv"
```

## Find Today's Runs
```bash
find outputs -name "run_*" -mtime 0
```

## View Latest Results
```bash
cat $(find outputs -name "detailed_summary.txt" | sort | tail -1)
```

## View Latest Plots
```bash
ls $(find outputs -type d -name "plots" | sort | tail -1)
```

## Compare Two Runs
```bash
diff outputs/run_AAAAAAAA_BBBBBB/results/algorithm_comparison.csv \
     outputs/run_CCCCCCCC_DDDDDD/results/algorithm_comparison.csv
```

## Clean Old Runs (Keep Last 5)
```bash
cd outputs
ls -t run_* | tail -n +6 | xargs rm -rf
```

## Archive Runs Older Than 30 Days
```bash
mkdir -p archived_runs
find outputs -name "run_*" -type d -mtime +30 -exec mv {} archived_runs/ \;
```

## Disk Usage Per Run
```bash
du -sh outputs/run_*/
```

## Count Total Runs
```bash
ls -d outputs/run_*/ | wc -l
```

---
**See TIMESTAMPED_OUTPUTS_GUIDE.md for complete documentation**
