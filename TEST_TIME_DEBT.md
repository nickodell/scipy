## Time Debt Tracking

Running total: -14.76s

### Changes

| Test | File | Change | Duration |
|------|------|--------|----------|
| test_resampling_2x2 | test_stats.py | promoted (slow→fast) | +1.73s |
| test_some_code_paths | test_stats.py | promoted (slow→fast) | +0.29s |
| test_resampling_pvalue[monte_carlo2-two-sided] | test_stats.py | demoted (fast→slow) | -2.11s |
| test_resampling_pvalue[monte_carlo2-greater] | test_stats.py | demoted (fast→slow) | -2.11s |
| test_resampling_pvalue[monte_carlo2-less] | test_stats.py | demoted (fast→slow) | -1.53s |
| test_bootstrap_ci | test_stats.py | promoted (slow→fast) | +3.11s |
| test_all_nograd_minimizers[COBYLA] | test__basinhopping.py | demoted (fast→slow) | -3.40s |
| test_spiral_cleanup | test_bary_rational.py | demoted (fast→slow) | -3.95s |
| test_woodbury | test_bsplines.py | demoted (fast→slow) | -3.79s |

Total: 1.73 + 0.29 - 2.11 - 2.11 - 1.53 + 3.11 - 3.40 - 3.95 - 3.79 = **-11.76s**
