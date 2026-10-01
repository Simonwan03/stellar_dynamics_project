# Generated Audit

Default dataset choice: `dataset_v2` / sampling `1`

## Discovered dataset candidates

### /home/wan/stellar_dynamics/dataset (sampling 0)

- result shards: 5
- local batch ids: [0, 1, 2, 3, 4]
- local result rows: 40000
- label path: `/home/wan/stellar_dynamics/dataset/completed_systems_sampling_0.txt`
- label rows: 40000
- invalid labels: 0
- time horizon max: 5000.0
- first shard time length: 1001
- first shard shapes:
  - `APs_complete`: `(8000, 1001, 2)`
  - `LANs_complete`: `(8000, 1001, 2)`
  - `angMomentum_complete`: `(8000, 1001, 3)`
  - `aratios_complete`: `(8000, 1001, 1)`
  - `eccs_complete`: `(8000, 1001, 2)`
  - `energy_complete`: `(8000, 1001, 1)`
  - `incs_complete`: `(8000, 1001, 2)`
  - `masses_complete`: `(8000, 1001, 3)`
  - `mratios_complete`: `(8000, 1001, 2)`
  - `periods_complete`: `(8000, 1001, 2)`
  - `positions_complete`: `(8000, 1001, 3, 3)`
  - `smaxes_complete`: `(8000, 1001, 2)`
  - `tAnos_complete`: `(8000, 1001, 2)`
  - `times_complete`: `(8000, 1001)`
  - `velocities_complete`: `(8000, 1001, 3, 3)`

### /home/wan/stellar_dynamics/dataset_v2 (sampling 1)

- result shards: 5
- local batch ids: [0, 1, 2, 3, 4]
- local result rows: 40000
- label path: `/home/wan/stellar_dynamics/dataset_v2/completed_systems_sampling_1.txt`
- label rows: 400000
- invalid labels: 139
- time horizon max: 10000.0
- first shard time length: 2001
- first shard shapes:
  - `APs_complete`: `(8000, 2001, 2)`
  - `LANs_complete`: `(8000, 2001, 2)`
  - `angMomentum_complete`: `(8000, 2001, 3)`
  - `aratios_complete`: `(8000, 2001, 1)`
  - `eccs_complete`: `(8000, 2001, 2)`
  - `energy_complete`: `(8000, 2001, 1)`
  - `incs_complete`: `(8000, 2001, 2)`
  - `masses_complete`: `(8000, 2001, 3)`
  - `mratios_complete`: `(8000, 2001, 2)`
  - `periods_complete`: `(8000, 2001, 2)`
  - `positions_complete`: `(8000, 2001, 3, 3)`
  - `smaxes_complete`: `(8000, 2001, 2)`
  - `tAnos_complete`: `(8000, 2001, 2)`
  - `times_complete`: `(8000, 2001)`
  - `velocities_complete`: `(8000, 2001, 3, 3)`
