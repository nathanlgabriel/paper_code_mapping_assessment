# Comprehensive Code-to-Paper Mapping & Holistic Model Architecture
## Corrected Version

> **About this document:** This document corrects the refined Qwen 35B mapping.
> All modifications are marked:
> - **[⚠ CORRECTED]** — a factual error has been fixed.
> - **[➕ ADDED]** — new material not present in the prior document.
> - **[❌ PRIOR ERROR]** — a brief inline label identifying the specific mistake.
>
> Unmarked text is carried over from the prior document and is not disputed.
> The prior document is substantially more accurate than earlier attempts, and most
> of its content is correct. The remaining errors cluster around a single persistent
> theme: legacy pair-based functions (`connections_init`, `pairings`) are still
> described as active components of the simulation.

---

## I. Holistic Execution Architecture & Data Flow

The code implements the paper's model as a discrete-time iterative simulation loop.
Each simulation run follows a strict, reusable pipeline:

1. **Initialization:** Urns/profiles are initialized with random one-hot distributions
   (replicator) or uniform `inertia=1` weights (RL, set in the notebook). Mixed-radix
   multipliers (`sigmultipliers`, `repmultipliers`) and dimension caps (`profilecaps`)
   are computed to map multidimensional strategy strings to integer indices.

2. **Assortment Weighting:**

   > **[⚠ CORRECTED — ❌ `pairings` described as part of active loop]**
   > The prior document stated: "Agents are paired via weighted sampling without
   > replacement (`pairings`), implementing Appendix A's homophily framework."
   > **`pairings` is not called by either active notebook.** The active assortment
   > mechanism works analytically: `execassort_array_create` builds a static weight
   > matrix, `find_assort_multipliers` normalizes it against current population signal
   > counts, and the resulting `assort_multipliers[i][j]` factor is applied directly
   > inside `executilcompute` / `executilcomputeREforget` to scale payoffs without
   > any explicit pair sampling.

   A similarity matrix is computed based on matching non-zero signal dimensions
   (`execassort_array_create`). Weights are normalized by population signaling
   disposition (`find_assort_multipliers`). Assortment weights are applied analytically
   inside utility/RL update functions, implementing Appendix A's homophily framework.

3. **Interaction & Learning:**
   - **Replicator:** Population-level utilities are computed against all present
     profiles (`executilcompute`). Frequencies are updated via the discrete replicator
     equation (`genBSreplicator_single_tstep`). Mutations are applied via selective
     component replacement (`repmutation_smart`).
   - **Reinforcement (Roth-Erev):** A single strategy profile is drawn per agent for
     the entire timestep (`random2picks`). Payoffs are computed pairwise, urns are
     reinforced, and a forgetting factor is applied (`executilcomputeREforget`).

4. **Monitoring & Conversion:** Continuous urns are converted back to discrete
   signals/actions/executive states for logging (`execrepconvert`, `execrepconvertRE`).
   Time-series, signal distributions, and attention thresholds are aggregated
   (`time_update`, `genBSexec_results`, `getalphasignaling`, `average_mutual_info`).

The two notebooks isolate the two learning dynamics while sharing identical payoff,
assortment, signaling, executive, and monitoring infrastructure. Data flows from
continuous urn distributions (RL) or integer profile arrays (Replicator) → analytical
assortment weighting → payoff/learning → back to discrete/continuous states for analysis.

---

## II. Core Model Components: Theory ↔ Code

### A. Generalized Bach/Stravinsky Game & Preferences

| Paper Concept        | Paper Reference   | Code Variable/Function                     | Implementation Detail                                                                 |
|:-------------------- |:----------------- |:------------------------------------------ |:------------------------------------------------------------------------------------- |
| Coordination payoffs | Tables 3, 6, 7, 8 | `coordination_preferencesf[type][action]`  | **2D array** [numtypes, numactions]. Passed into simulation. Maps preference type to action payoff. Matches paper exactly. |
| Failure punishment   | Sec 2.1, 3.3      | `genBSpunishf`                             | Applied when actions mismatch. Set to `0` in active sweeps. Matches payoff tables.        |
| Net payoff           | Sec 2.1, 3.3      | `coord_pref + sigcost + assortment_weight` | Computed in `executilcompute` (Replicator) and `executilcomputeREforget` (RL).        |

### B. Multidimensional Signaling & Executive/Attention Dynamics

| Paper Concept          | Paper Reference | Code Variable/Function          | Implementation Detail                                                                                                                                                         |
|:---------------------- |:--------------- |:------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signaling dimensions   | Sec 4.1         | `sigdimensionsf`                | Number of independent signaling axes (K).                                                                                                                                     |
| Signals per dimension  | Sec 4.1         | `sigperdimf`                    | e.g., `[2, 2]` → signals 0,1 in each dim.                                                                                                                                     |
| Signal multipliers     | Sec 4.1         | `sigmultipliersf`               | Mixed-radix weights to flatten multidimensional signals into single indices.                                                                                                  |
| Null signal (ignoring) | Sec 3.3, 4.1    | Signal position 0 in profile / urn; null-masking in `executilcompute`/`executilcomputeREforget` | If signal is 0 in a dimension, cost is 0, actions don't condition on it. Implemented by zeroing any dimension where either agent has a 0 in that dim, inside the payoff loop. **[⚠ CORRECTED below — `execurnsf` does not gate simulation dynamics]** |
| Signal cost            | Sec 3.3         | `sigcostf`                      | Negative value (e.g., `-0.000125` or `-0.0005`). Applied per attended dimension.                                                                                              |

> **[⚠ CORRECTED — ❌ `execurnsf` described as gating simulation-time attention]**
> The prior document stated: "If dimension attended (`execurnf[d]==1`), agents signal
> in `[1..sigperdimf[d]]`. If unattended (`execurnf[d]==0`), signal is `0`..."
> This implies `execurnsf` is read during the simulation to control agent behavior.
> **`execurnsf` is populated only at reporting time** — by `execrepconvert` (replicator)
> or `execrepconvertRE` (RL) — via `argmax` of the signal urns. It is not read during
> `executilcompute` or `executilcomputeREforget`. The actual null-signal attention
> mechanism during simulation operates by checking whether a profile's signal position
> is 0 directly inside the payoff loop, without reference to `execurnsf`.

### C. Assortative Pairing & Homophily

| Paper Concept                 | Paper Reference     | Code Variable/Function         | Implementation Detail                                                                                        |
|:----------------------------- |:------------------- |:------------------------------ |:------------------------------------------------------------------------------------------------------------ |
| Homophily parameter           | Appendix A Eq. 1-2  | `homophily_factorf`            | `h`. `h=0` → random mixing. `h=1` → double weight per shared non-null signal dimension.                      |
| Base connection weights       | Appendix A          | `base_connection_weightsf`     | `[1, 2, 4, 8, 16, 32]`. Entry `d` = 2^d, used as S(h,x,j) = (2^d)^h.                                        |
| Assortment utility `H(h,x,i)` | Appendix A          | `find_assort_multipliers`      | Computes `H = [N / ΣS] × S`. Normalization prevents utility inflation when all agents use identical signals. |
| Assortment weight matrix      | Appendix A          | `execassort_array_create`      | Builds static [numsignals × numsignals] weight matrix. Counts only **non-null** matching dimensions.         |

> **[⚠ CORRECTED — ❌ `connections_init` and `pairings` listed as active assortment functions]**
> The prior document listed `connections_init` and `pairings` under "Weighted pairing"
> as implementing Appendix A's homophily framework. **Neither function is called by
> either active notebook.** They belong to legacy pair-based simulation variants that
> were superseded. The active assortment implementation applies H(h,x,i) analytically
> via `assort_multipliers[i][j]` inside `executilcompute` and `executilcomputeREforget`
> — there is no explicit pair-drawing step in the active model. See also the corrected
> inactive functions list in Section VII.

---

## III. Learning Dynamics: Divergence & Shared Infrastructure

### A. Deterministic Replicator Dynamics (Notebook 2)

| Paper Concept           | Paper Reference | Code Function                  | Implementation Detail                                                                                                                   |
|:----------------------- |:--------------- |:------------------------------ |:--------------------------------------------------------------------------------------------------------------------------------------- |
| Replicator equation     | Sec 2.2 Eq. 1   | `genBSreplicator_single_tstep` | `N(x) += N(x) × (U(x) - Avg(U_k))`. Computes `Avg(U_k)` across present profiles. Sets negatives to 0.                                   |
| Utility calculation     | Sec 2.2 Eq. 2   | `executilcompute`              | Sums payoffs across all present profiles `i`, weighted by `M(i)` and `H(h,x,i)`. Self-interaction uses `M(i)-1`. Matches paper exactly. |
| Frequency normalization | Sec 2.2         | `genBSreplicator_single_tstep` | Scales updated frequencies so type counts remain constant.                                                                              |
| Cascade rounding        | Not in paper    | `genBSreplicator_single_tstep` | Converts fractional frequencies to integer agent counts without systematic bias across profiles. Computational bridge.                  |
| Zero-correction         | Not in paper    | `flip7zero_correction`         | Reinitializes vanishing strategy types with random profiles to prevent numerical collapse.                                              |

### B. Reinforcement Learning (Roth-Erev) with Forgetting (Notebook 1)

| Paper Concept            | Paper Reference | Code Function             | Implementation Detail                                                                                                             |
|:------------------------ |:--------------- |:------------------------- |:--------------------------------------------------------------------------------------------------------------------------------- |
| Agent-level urns         | Appendix B      | `sigurnsf`, `recurnsf`    | Sender and receiver urn arrays. Continuous counts act as action weights. **Initialized to `inertia=1` in the notebook, not by `RothErevExec_initialize`.** |
| Single-profile execution | Appendix B      | `random2picks`            | Draws one complete strategy profile per agent for the entire timestep. Reduces noise, improves tractability.                      |
| Reinforcement rule       | Appendix B      | `executilcomputeREforget` | Adds payoff amount to **only the urn positions that were used** in the interaction. `sigurnsf` reinforced for broadcast signals; `recurnsf` for chosen actions. |
| Forgetting factor        | Appendix B      | `forgetf` (e.g., `0.998`) | Multiplies **all** urn counts at the **start** of each timestep's update (before reinforcement). Floors at `1` to preserve optionality. |
| Epsilon smoothing        | Not in paper    | `epsilonf`                | Added to uniform random draws to guarantee `(0,1]` sampling, preventing zero-probability states.                                  |

> **[➕ ADDED — Forgetting order]** Forgetting (multiply by φ) is applied to all urn
> entries at the **beginning** of `executilcomputeREforget`, before any reinforcement
> is added for the current timestep's interactions. This ordering is not stated explicitly
> in Appendix B but is what the code implements, and it is consequential for dynamics.

---

## IV. State Representation & Mixed-Radix Indexing

The paper describes strategy profiles as strings of length `K + θ` where K = signal
dimensions and θ = number of possible signal combinations (the action-response positions).

> **[⚠ CORRECTED — ❌ Profile string described as including "executive attention" as separate component]**
> The prior document stated: "strings of length `K + θ` (signal dimensions + signal
> combinations + executive attention)." Executive attention is **not a separate component**
> of the profile string. The profile string has exactly two types of positions: K signal
> positions (first K entries) and θ action positions (next θ entries). Executive attention
> (attending vs. not attending) is inferred from whether a signal position is 0 (not
> attending) or non-zero (attending), not stored as a separate third component.

| Code Variable                   | Paper Correlation                 | Computational Bridge                                                                                |
|:------------------------------- |:--------------------------------- |:--------------------------------------------------------------------------------------------------- |
| `repmultipliersf`               | Mixed-radix weights for full profile | Positional weights converting full profile string (signals + actions) to a unique integer index.   |
| `profilecaps`                   | Valid range per profile position  | Entry i = number of options at position i. Used in mutation and `random2picks` to bound draws.      |
| `profiles_indexed`              | Strategy space enumeration        | Precomputes all valid `numprofilesf` profiles as integer arrays. Enables efficient utility lookups. |
| `agentprofiles` (Replicator)    | Agent strategy mapping            | Integer array mapping each agent to a profile index. Rebuilt each timestep.                         |
| `agentprofilepicks` (RL)        | Current strategy per agent        | Array holding drawn signal/action values for the timestep. Overwritten each timestep by `random2picks`. |
| `execrepconvert`                | Discrete → Urn mapping (Replicator) | Converts integer profiles back to one-hot `sigurnsf`/`recurnsf`/`execurnsf` for logging. **Active in replicator only.** |
| `execrepconvertRE`              | Stochastic urn → Binary attention (RL) | Extracts `argmax(sigurnsf)` to determine deterministic attention state (`0` or `1`) for analysis. |

> **[⚠ CORRECTED — ❌ `repconvert` listed alongside `execrepconvert` as active]**
> The prior document listed "`repconvert` / `execrepconvert`" together. `repconvert`
> (without the "exec" prefix) is a legacy non-exec function not called by the active
> replicator notebook. Only `execrepconvert` is active.

---

## V. Evolutionary Mechanics & Numerical Stability

| Mechanism                      | Paper Reference | Code Implementation                                     | Purpose / Gap Filled                                                                                              |
|:------------------------------ |:--------------- |:------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------- |
| Agent-level mutation prob `m0` | Sec 2.2         | `mutateratef[0]`                                        | Probability an agent's profile is selected for mutation.                                                          |
| Component flip prob `m1`       | Sec 2.2         | `mutateratef[1]`                                        | Probability a specific profile position is replaced with a random alternative.                                    |
| Mutation mask generation       | Sec 2.2         | `muterandoms10_smart`, `numbagreater`, `numbagreater3d` | Pre-generates boolean masks. Batch of 10 rounds; refreshed every 10 timesteps.                                    |
| Smart mutation application     | Sec 2.2         | `repmutation_smart`                                     | Position-by-position profile mixing: uses random value with prob m1, keeps old value otherwise. Updates counts.   |
| Cascade rounding               | Not in paper    | `genBSreplicator_single_tstep`                          | Converts fractional frequencies to integer agent counts without bias via cumulative `np.around`.                  |
| Urn floor at 1                 | Appendix B      | `executilcomputeREforget`                               | Ensures strategies never become permanently extinct due to stochastic forgetting. Applied after reinforcement.    |
| Pre-generated RNG arrays       | Not in paper    | `picks10`, `mutations10`, `muteprofilestrings10`        | Avoids repeated RNG calls inside Numba JIT loops; batch generation every 10 timesteps.                           |
| **Mutation absent from RL**    | Appendix B      | RL main loop                                            | **[➕ ADDED]** `mutateratef` is passed to the RL function but not used. Exploration is provided by stochastic urn sampling and the urn floor constraint instead. |

---

## VI. Monitoring, Structural Quantification & Analysis

| Function                                | Paper Correlation | Purpose & Computational Bridge                                                                                                                                                                                                                                           |
|:--------------------------------------- |:----------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `time_update`                           | Sec 4.2           | Logs normalized action distributions per type/signal over time. Fills trajectory tracking gap for convergence analysis.                                                                                                                                                  |
| `getalphasignaling`                     | Sec 3.5, 4.2      | Computes empirical "alpha" signals (most frequent per dim by type), attention thresholds (`>0.5` agreement), opposition/agreement counts. **Also relabels signals consistently across runs so equivalent equilibria with different arbitrary signal labels can be aggregated.** |
| `average_mutual_info`                   | Sec 4.3           | Computes Shannon MI between signals and actions. Proxy for coordination efficiency/synergy. Fills information-theoretic metric gap (paper mentions PIDE/synergy but code uses MI as robust, computable proxy).                                                           |
| `genBSexec_results` / `genBS_resultsRE` | Sec 3.5, 4.3      | Aggregates most frequent signals/actions per type. Handles attention masking: if agent doesn't attend a dimension, that dimension's signal is zeroed out for interaction interpretation. Matches paper's distinction between signaling systems and behavioral responses. |

> **[➕ ADDED — Convergence check]** Both active notebooks include a convergence check
> that halts the simulation early if the population has stabilized:
> - **Replicator:** every 100 timesteps, compares `agentprofilesCOPY` to `agentprofiles`.
> - **RL:** every 1,000 timesteps, compares `agentprofilepicksCOPY` to `agentprofilepicks`.
> The longer RL interval reflects the fact that stochastic urn sampling produces different
> `agentprofilepicks` on each timestep even from stable urns, making a 100-step check
> unreliable.

---

## VII. Exhaustive Function-to-Paper Variable Mapping (Used Functions Only)

### `genBS_f7_RothErevExec_full_play_typed` (RL Entry Point)

| Variable                                  | Type       | Paper Correlation                                                      |
|:----------------------------------------- |:---------- |:---------------------------------------------------------------------- |
| `forgetf`                                 | float      | Appendix B: φ (0.998). Forgetting multiplier.                          |
| `mutateratef`                             | ndarray    | Sec 2.2: [m0, m1]. **Not used in RL path** but passed for compatibility. |
| `sigcostf`                                | float      | Sec 3.3: Cost per attended dimension.                                  |
| `repmultipliersf`                         | ndarray    | Mixed-radix weights for profile indexing.                              |
| `numprofilesf`                            | int        | Total strategy space size.                                             |
| `signal_snapshotsf`, `record_intervalf`   | int        | Monitoring intervals.                                                  |
| `genBSpunishf`                            | ndarray    | Sec 2.1/3.3: Failure payoff (= 0 in active sweeps).                   |
| `numtypesf`, `numsignalsf`, `numactionsf` | int        | Tables/Sec 2.1: numtypes, θ, numactions.                              |
| `coordination_preferencesf`               | ndarray    | Tables 3,6,7,8: p_{xi,k}. **2D array [numtypes, numactions].** Divided by 200 in RL notebook to match Appendix B Table 1 magnitudes. |
| `poptf`                                   | ndarray    | Sec 2.2: Agent preference types.                                       |
| `sigurnsf`, `recurnsf`                    | ndarrays   | Appendix B: Sender/receiver urns. Initialized to `inertia=1` in notebook. |
| `population_sizef`                        | int        | Sec 2.2: N (500 in RL notebook, 1000 in replicator).                  |
| `sigdimensionsf`                          | int        | Sec 4.1: K, number of signaling axes.                                  |
| `base_connection_weightsf`                | ndarray    | Appendix A: Base weights per match count. [1,2,4,8,16,32].            |
| `sigmultipliersf`                         | ndarray    | Sec 4.1: Signal index multipliers.                                     |
| `homophily_factorf`                       | float      | Appendix A: h (0 in RL notebook, 1 in replicator notebook).           |
| `rngf`                                    | RNG object | Sec 2.2/Appendix B: Random state.                                     |
| `epsilonf`                                | float      | Computational bridge: Zero-probability prevention.                     |
| `runid`                                   | int        | Computational bridge: Run tracking.                                    |

### `genBSreplicatorexec_full_play_typed` (Replicator Entry Point)

Identical parameters to RL entry point except `mutateratef` is **actively used** in the
replicator path (m0 and m1 applied by `muterandoms10_smart`/`repmutation_smart`). `forgetf`
is not used in the replicator. `coordination_preferencesf` uses raw values (not divided by
200).

### Core Simulation & Utility Functions

| Function                                                                     | Key Variables                                                                                                                                                                                                                                                                                        | Paper Correlation & Purpose                                                                                                                           |
|:---------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RothErevExec_initialize`                                                    | `epsilonf`, `rngf`, `repmultipliersf`, `numprofilesf`, `numsignalsf`, `numactionsf`, `numtypesf`, `sigdimensionsf`, `sigurnsf`, `recurnsf`, `population_sizef`                                                                                                                                       | Appendix B: Computes `sigperdimf`/`profilecaps` only. **Does NOT initialize urn contents** — urns are pre-initialized to `inertia=1` in the notebook. **[⚠ CORRECTED — prior doc said "Initializes one-hot urns"]** |
| `replicatorexec_initialize`                                                  | Same as above                                                                                                                                                                                                                                                                                        | Sec 2.2: Initializes one-hot urns, precomputes `profiles_indexed`, `sigperdim`, `profilecaps`.                                                        |
| `genBSreplicatorexec_k_step`                                                 | `numtypesf`, `rngf`, `population_sizef`, `numprofilesf`, `poptf`                                                                                                                                                                                                                                     | Sec 2.2: Assigns random initial profiles, computes `typestotal`, `profile_count_typed/untyped`, `agentprofiles`.                                      |
| `random2picks`                                                               | `randoms4pick`, `sigdimensionsf`, `sigurnsf`, `recurnsf`, `population_sizef`, `agentprofilepicks`, `profile_length`, `profilecaps`, `numsignalsf`                                                                                                                                                    | Appendix B: Draws single strategy profile per agent for timestep. Uses pre-generated `picks10` RNG arrays refreshed every 10 timesteps.               |
| `get_sig_countsRE`                                                           | `numsignalsf`, `agentprofilepicks`, `population_sizef`, `sigmultipliersf`, `sigdimensionsf`                                                                                                                                                                                                          | Appendix A: Aggregates signal counts from `agentprofilepicks` for assortment weighting (RL path).                                                     |
| `execassort_array_create`                                                    | `homophily_factorf`, `sigperdimf`, `numsignalsf`, `sigdimensionsf`, `base_connection_weightsf`                                                                                                                                                                                                       | Appendix A: Builds static similarity matrix. Counts **non-null** matching dimensions only.                                                            |
| `find_assort_multipliers`                                                    | `assort_array`, `assort_sig_counts`, `numsignalsf`, `population_sizef`                                                                                                                                                                                                                               | Appendix A: Computes `H(h,x,i) = [N/ΣS] × S`. Normalizes weights by current signal distribution.                                                     |
| `get_sig_counts`                                                             | `numsignalsf`, `profiles_indexed`, `numprofilesf`, `sigmultipliersf`, `sigdimensionsf`, `profile_count_untyped`                                                                                                                                                                                      | Appendix A: Aggregates signal counts from profile frequencies (Replicator path).                                                                      |
| `pairings`                                                                   | —                                                                                                                                                                                                                                                                                                    | **[⚠ CORRECTED — NOT ACTIVE]** `pairings` is a legacy function not called by either active notebook. Assortment is applied analytically, not via explicit pair-sampling. |
| `executilcompute`                                                            | `assort_multipliers`, `sigcostf`, `sigperdimf`, `numtypesf`, `sigmultipliersf`, `sigdimensionsf`, `numsignalsf`, `numprofilesf`, `profiles_indexed`, `profile_count_untyped`, `coordination_preferencesf`, `base_connection_weightsf`, `homophily_factorf`, `genBSpunishf`                           | Sec 2.2 Eq. 2/3.3: Computes `U_k(x)` across all present profiles, weighted by `M(i)` and `H(h,x,i)`. Self-interaction uses `M(i)-1`. Adds `sigcostf`. |
| `executilcomputeREforget`                                                    | `forgetf`, `agentprofilepicks`, `assort_multipliers`, `sigcostf`, `sigperdimf`, `sigmultipliersf`, `sigdimensionsf`, `numsignalsf`, `numactionsf`, `population_sizef`, `poptf`, `sigurnsf`, `recurnsf`, `coordination_preferencesf`, `base_connection_weightsf`, `homophily_factorf`, `genBSpunishf` | Appendix B: (1) Applies forgetting (× φ to all urns, **before** reinforcement). (2) Computes pairwise payoffs for all agent pairs. (3) Reinforces only used urn positions. (4) Floors all entries at 1. |
| `genBSreplicator_single_tstep`                                               | `numprofilesf`, `numtypesf`, `typestotal`, `profile_count_typed`, `agentprofiles`, `profile_utility_bytype`, `rngf`, `population_sizef`, `poptf`                                                                                                                                                     | Sec 2.2 Eq. 1: Implements replicator update, averages utility, normalizes, applies cascade rounding, calls `flip7zero_correction`.                    |
| `time_update`                                                                | `typed_time`, `signal_time`, `typed_time_norm`, `signal_time_norm`, `poptf`, `sigurnsf`, `recurnsf`, `population_sizef`, `numsignalsf`, `numactionsf`, `numtypesf`, `sigdimensionsf`, `sigmultipliersf`, `timedex`                                                                                   | Sec 4.2/3.5: Logs normalized action distributions per type/signal over time.                                                                          |
| `getalphasignaling`                                                          | `popt`, `final_execurns`, `type0`, `type0opposition`, `type0agree`, `sigmultipliers`, `population_size`, `final_sigurns`, `numsignals_perdim`, `sigdimensions`, `runs`, `numtypes`, `numsignals`                                                                                                     | Sec 3.5/4.2: Computes empirical alpha signals, attention thresholds, opposition/agreement counts. **Critically, also relabels signals consistently across runs for cross-run aggregation.** |
| `average_mutual_info`                                                        | `numsignalsf`, `sigdimensionsf`, `sigmultipliersf`, `numactionsf`, `sigurnsf`, `recurnsf`, `population_sizef`, `runsf`                                                                                                                                                                               | Sec 4.3: Computes Shannon MI between signals and actions as coordination efficiency proxy.                                                            |
| `flip7zero_correction`                                                       | `slice_type`, `profile_count_typed_slice`, `numtypesf`, `rngf`, `population_sizef`, `numprofilesf`, `poptf`                                                                                                                                                                                          | Computational bridge: Reinitializes vanishing strategy types to prevent numerical collapse.                                                           |
| `muterandoms10_smart`, `numbagreater`, `numbagreater3d`, `repmutation_smart` | `rngf`, `mutateratef`, `population_sizef`, `numprofilesf`, `stringlength`, `repmultipliersf`, `profilecaps`, `profiles_indexed`, `numtypesf`, `numprofilesf`, `profile_count_typed`, `profile_count_untyped`, `mutations`, `newprofiles`, `muteprofilestrings`                                       | Sec 2.2: Implements `m0`/`m1` mutation protocol. Pre-generates masks, applies selective component replacement, updates counts. **Replicator only; not called in RL path.** |
| `execrepconvert`                                                             | `sigurnsf`, `recurnsf`, `execurnsf`, `agentprofiles`, `profiles_indexed`, `sigdimensionsf`, `numsignalsf`, `population_sizef`                                                                                                                                                                        | Sec 4.1/3.3: Maps integer profiles back to one-hot urns and infers binary executive states for logging (Replicator). |
| `execrepconvertRE`                                                           | `sigurnsf`, `recurnsf`, `execurnsf`, `sigdimensionsf`, `numsignalsf`, `population_sizef`                                                                                                                                                                                                             | Sec 4.1/3.3: Extracts deterministic attention from stochastic urns via `argmax`. Both `execurnsf` outputs are **reporting-time only**, not used during simulation dynamics. |
| `genBSexec_results`                                                          | `snapcount`, `sigperdimf`, `profiles_indexed`, `agentprofiles`, `numactionsf`, `poptf`, `sigdimensionsf`, `sigmultipliersf`, `numtypesf`, `numsignalsf`, `sigurnsf`, `recurnsf`, `execurnsf`, `population_sizef`                                                                                     | Sec 3.5/4.3: Aggregates signals/actions per type. Computes `simple_stat_001` (argmax) and `simple_stat_002` (>75% threshold).                        |
| `genBS_resultsRE`                                                            | `sigperdimf`, `numactionsf`, `poptf`, `sigdimensionsf`, `sigmultipliersf`, `numtypesf`, `numsignalsf`, `sigurnsf`, `recurnsf`, `population_sizef`                                                                                                                                                    | Sec 3.5/4.3: Same aggregation as `genBSexec_results` for RL path.                                                                                   |

> **[⚠ CORRECTED — Inactive function list is incomplete]**
> The prior document's inactive note listed: "`genBS_check_success`, `genBS_single_tstep`,
> `assort_array_create`, `utilcompute`, `expandedsignals`, `genBS_results`, `greetings_*`,
> `muterandoms10`, `repmutation`, `numbagreater` (2D), `repconvert`."
> This list should also include: **`connections_init`**, **`pairings`** (active assortment
> is computed analytically, not via these functions), **`executilcomputeRE`** (RL updater
> without forgetting, present in the RL script but superseded by `executilcomputeREforget`),
> **`genBSreplicator_first_tstep`** (commented out in active code; replaced by
> `genBSreplicatorexec_k_step`), **`genBSreplicator_full_play`** (non-exec main loop),
> **`replicator_initialize`** (non-exec init), **`social_draws`**, **`receiver_draws`**.
>
> *Corrected inactive list:* `genBS_check_success`, `genBS_single_tstep`,
> `assort_array_create`, `utilcompute`, `expandedsignals`, `genBS_results`, `greetings_*`,
> `muterandoms10`, `repmutation`, `numbagreater` (2D), `repconvert`, **`connections_init`**,
> **`pairings`**, **`executilcomputeRE`**, **`genBSreplicator_first_tstep`**,
> **`genBSreplicator_full_play`**, **`replicator_initialize`**, **`social_draws`**,
> **`receiver_draws`**.

---

## VIII. Synthesis: Fidelity, Bridges, & Theoretical Realization

The code is a high-fidelity, computationally optimized implementation of the paper's
theoretical framework. Every mathematical equation, table, appendix specification, and
narrative mechanism has a direct computational counterpart:

1. **Payoff & Utility Architecture:** The paper's generalized BoS payoff matrices
   (`coordination_preferencesf`) are directly mapped to 2D array lookups [type][action].
   Failure payoffs (`genBSpunishf`, = 0 in active sweeps) and signal costs (`sigcostf`,
   negative, applied per active dimension) are applied exactly as described. Assortment
   utility `H(h,x,i)` from Appendix A is implemented with explicit normalization to
   prevent utility inflation, matching the supplement's mathematical requirement.

2. **Learning Dynamics Isolation:** The dual-dynamic design cleanly separates replicator
   (population-level frequency update + cascade rounding) from reinforcement learning
   (agent-level urn reinforcement + forgetting + single-profile execution). This matches
   the paper's explicit comparison in Section 2.2 vs Appendix B, where RL is noted as
   advantageous for future dynamic preference modeling.

3. **Multidimensional Signaling & Executive Attention:** The code implements the paper's
   K-dimensional signaling framework precisely. The null-signal attention mechanism
   (zeroing any dimension where either agent has signal 0) is applied inside
   `executilcompute`/`executilcomputeREforget` directly. Binary executive states
   (`execurnsf`) are inferred at **reporting time** from urn argmax and gate analysis,
   not simulation-time dynamics.

4. **Computational Bridges Filling Theoretical Gaps:** Where the paper omits
   implementation details, the code provides explicit, well-documented solutions:
   - **Mixed-radix indexing** (`repmultipliers`, `profilecaps`, `profiles_indexed`)
     converts theoretical strategy strings to efficient integer arrays.
   - **Cascade rounding** converts fractional replicator frequencies to integer agent
     counts without bias.
   - **Epsilon smoothing** and **urn floor at 1** prevent numerical collapse in stochastic
     environments.
   - **Pre-generated RNG arrays** avoid repeated RNG overhead inside Numba JIT loops.
   - **Zero-correction** reinitializes vanishing strategies to maintain population integrity.
   - **Forgetting before reinforcement**: within `executilcomputeREforget`, all urns are
     decayed by φ first, then payoffs are added on top of decayed values.

5. **Monitoring & Structural Quantification:** The paper describes equilibrium outcomes
   but does not specify trajectory tracking. The code fills this with `time_update` for
   normalized logging, `getalphasignaling` for empirical alpha signal computation and
   cross-run signal relabeling, and `average_mutual_info` for coordination efficiency
   measurement.

6. **Execution Pipeline Fidelity:** The active notebook-to-script call chain strictly
   follows the paper's loop: `initialize → assortment weights (analytical) → draw
   profile/compute utility → learn/update → convert/monitor`. Assortment weighting,
   payoff calculation, and result aggregation are shared across both dynamics, confirming
   that only the learning mechanism diverges, exactly as intended by the paper's design.

**Conclusion:** The code implements the paper's model with mathematical equivalence and
computational rigor. Every equation, table, appendix specification, and narrative mechanism
is explicitly realized. The primary correction to this document is the removal of
`connections_init` and `pairings` from the active simulation path — the actual assortment
mechanism is entirely analytical, embedded within the utility and RL update functions.
