# Open scientific definition checks

## Task C: events exactly on an interior hazard-bin boundary

Status: open; existing labels, training code, predictions, and metrics are unchanged.

In `tasks/task_c/survival.py`, `build_discrete_hazard_targets` uses intervals `(edge[k], edge[k+1]]`, so an event with `time_unbound == edge[k+1]` is assigned to the bin ending at that edge. `survival_targets_from_times` uses `time_unbound >= upper_edge`, so the empirical survival target still counts that same variation as surviving at the edge. The two targets disagree at an interior edge; a consistent convention needs an explicit scientific decision before changing either definition.

For the current sampling 3 complete-system subset (159,936 trajectories from 4,998 systems), `time_unbound` equals interior edges 10, 100, and 1000 for 991, 161, and 23 trajectories respectively (1,175 total). The numerical effect on reported Task C metrics has not been measured. Reproducing the saved experiments retains the existing convention until this issue is resolved.
