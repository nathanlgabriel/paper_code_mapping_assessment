# Code–Paper Correspondence: Social Identity Signaling Simulations
## Corrected Review of Fifth-Model Attempt (Nemotron 30B)

> **About this document:** This document corrects and supplements the fifth model's mapping.
> All modifications are marked:
> - **[⚠ CORRECTED]** — a factual error has been fixed.
> - **[➕ ADDED]** — new material not present in the prior document.
> - **[❌ PRIOR ERROR]** — a brief inline label identifying the specific mistake.
>
> **Overall note:** This document contains the most severe identification errors of any
> reviewed so far. The dynamics labels in Section 1 are completely reversed, and this
> inversion propagates throughout Sections 2 and 4, producing large blocks of detailed
> description that are attributed to the wrong notebook. Additionally, several legacy
> functions are described as part of the active simulation loop. Corrections are extensive.

---

## 1. Which Notebook Implements Which Learning Dynamics?

> **[⚠ CORRECTED — ❌ CRITICAL: Dynamics labels are completely reversed]**
> The prior document assigns:
> - `genBS_v0055k05_BZforget3...ipynb` → **Replicator dynamics**
> - `genBS_v0055k_assort_FLIP7...ipynb` → **Reinforcement/Roth-Erev dynamics**
>
> **This is exactly backwards.** The BZforget3 notebook IS the Roth-Erev RL notebook;
> the FLIP7 notebook IS the replicator notebook. The function names embedded in each
> notebook's filename and each script's filename make this unambiguous:
> - `BZforget3` = "forgetting" mechanism (RL); `RothErevExec` = Roth-Erev RL
> - `FLIP7_repEXEC` = replicator exec; `SMARTmutation` = replicator mutation
>
> The prior document also produced incorrect function lists for each notebook in its
> identification key, and this reversal propagates throughout Sections 2.1, 2.2, and 4.
> The corrected identification is below.

| Notebook | Script | Learning Dynamic |
|:---|:---|:---|
| **`genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`** | `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py` | **Roth-Erev Reinforcement Learning with Forgetting** (Supplement Appendix B). The main loop calls `genBS_f7_RothErevExec_full_play_typed`, which uses `random2picks`, `executilcomputeREforget`, `get_sig_countsRE`, and `find_assort_multipliers`. |
| **`genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`** | `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py` | **Discrete Replicator Dynamics** (Paper Section 2.2). The main loop calls `genBSreplicatorexec_full_play_typed`, which uses `executilcompute`, `genBSreplicator_single_tstep`, `muterandoms10_smart`, and `repmutation_smart`. |

> **[➕ ADDED]** Each notebook imports from its **own distinct Python script**. The two
> scripts share analysis and assortment utility functions but contain separate implementations
> of their respective learning dynamics.

---

## 2. Mapping Every Subfunction to the Description in the Paper / Supplement

> **[⚠ CORRECTED — ❌ Sections 2.1 and 2.2 are swapped throughout]**
> Because the prior document reversed the dynamics labels, its Section 2.1 ("Notebook A –
> Replicator Dynamics") actually describes functions from the **RL** notebook/script, and
> its Section 2.2 ("Notebook B – Reinforcement Roth-Erev Dynamics") describes functions
> attributed to the **replicator** notebook. The corrected sections below swap and correct
> these attributions. Within each section, individual function-level errors are also noted.

---

### 2.1 Replicator Dynamics Notebook — `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep...ipynb`

| Code Function | Paper Passage | Corrected Implementation Notes |
|:---|:---|:---|
| `genBSreplicatorexec_full_play_typed` (top-level driver) | Section 2.2: "model dynamics proceeded in discrete timesteps using replicator dynamics" | Orchestrates the full simulation. Calls `replicatorexec_initialize`, then `genBSreplicatorexec_k_step` for random initialization, then the main loop: `get_sig_counts` → `find_assort_multipliers` → `executilcompute` → `genBSreplicator_single_tstep` → (every 10 steps) `muterandoms10_smart` / `repmutation_smart`. Checks convergence every 100 timesteps. |
| `replicatorexec_initialize` | Sec 2.2: "randomly assigned a behavior strategy" | Sets up `profiles_indexed`, `sigurnsf`, `recurnsf` with random one-hot assignments. Also sets up `execurnsf` structure. |
| `genBSreplicatorexec_k_step` | Sec 2.2: "randomly assigned a behavior strategy" | **The active random profile initializer.** Assigns each agent a uniformly random integer profile index. Produces initial `profile_count_typed`, `profile_count_untyped`, `agentprofiles`. |
| `execassort_array_create` | Appendix A: H(h,x,i) assortment formula | Builds static [numsignals × numsignals] assortment weight lookup. Counts only dimensions where **both** agents have non-null signals. |
| `find_assort_multipliers` | Appendix A: H(h,x,i) normalization | Normalizes raw assortment weights by population signal counts to produce H(h,x,i) for every signal pair. Called each timestep. |
| `get_sig_counts` | Implementation detail | Tallies how many agents broadcast each combined signal index; provides the denominator for assortment normalization. |
| `executilcompute` | Sec 2.2: utility equation U_k(x) | Computes expected payoff for every strategy profile × type pair. Applies assortment multipliers, coordination preferences, and signal cost per active dimension. |
| `genBSreplicator_single_tstep` | Sec 2.2: replicator equation | Updates profile counts via N_{t+1}(x) = N_t(x) + N_t(x) × [U_k(x) − Avg(U_k)]. Enforces non-negativity, renormalizes, performs cascade integer rounding, rebuilds `agentprofiles`. |
| `muterandoms10_smart` / `repmutation_smart` | Sec 2.2: mutation parameters m₀, m₁ | Two-parameter mutation: m₀ selects agents, m₁ governs per-position element replacement. Pre-generates 10 rounds for efficiency; applied every 10 timesteps. |
| `flip7zero_correction` | Not in paper | Numerically corrects the rare case where cascade rounding produces zero agents for a type, by reinitializing profiles randomly. |
| `execrepconvert` | Implementation detail | Converts integer profile counts back to urn format for reporting. `execurnsf` attention flags are inferred from signal urn argmax at this point. |
| `genBSexec_results` | Sec 4.2 optimality criteria | Computes `simple_stat_001` (modal actions) and `simple_stat_002` (>75% threshold) for sweep classification. |
| `time_update` | Table 4 | Records time-series of conditional action probabilities at each snapshot interval. |

> **[➕ ADDED — Inactive functions in the replicator script]** The following are present in
> the FLIP7 script but NOT called by the active replicator notebook: `utilcompute` (lacks
> signal cost), `replicator_initialize` (non-exec init), `reconvert` (non-exec conversion),
> `genBSreplicator_full_play` (non-exec loop), `genBSreplicator_first_tstep` (commented out),
> `connections_init`, `pairings`, `social_draws`, `receiver_draws`, `genBS_check_success`.

---

### 2.2 Reinforcement Learning Notebook — `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep...ipynb`

> **[⚠ CORRECTED — ❌ The prior document's Section 2.2 described the wrong active loop]**
> The prior document described the RL timestep as calling `social_draws` → `pairings` →
> `receiver_draws` → `greetings_check_success`. **None of these functions are called by
> `genBS_f7_RothErevExec_full_play`.** They belong to legacy pair-based variants. The
> correct active RL loop per timestep is described below.

| Code Function | Paper Passage | Corrected Implementation Notes |
|:---|:---|:---|
| `genBS_f7_RothErevExec_full_play_typed` (top-level driver) | Appendix B: "select a single strategy profile for all interactions per timestep" | Orchestrates the RL simulation. Per timestep: `random2picks` → `get_sig_countsRE` → `find_assort_multipliers` → `executilcomputeREforget`. Checks convergence every 1,000 timesteps. |
| `RothErevExec_initialize` | Appendix B: urn structure | Computes structural parameters `sigperdimf` and `profilecaps` only. Does NOT initialize urn contents. |
| Urn initialization to `inertia=1` | Appendix B: "one ball per action per urn" | Done in the **notebook itself** before the simulation call: `sigurns = [np.ones(...) * inertia]`, `recurns = np.ones(...) * inertia`. |
| `random2picks` | Appendix B: "drawing one ball from each urn at random" | Draws one signal per dimension and one action per signal combo from each agent's urns. Stores the result in `agentprofilepicks`. Called at the **start** of each timestep, before interactions. |
| `get_sig_countsRE` | Implementation detail | Tallies signal distribution from `agentprofilepicks` (not from profile counts). Used for assortment normalization. |
| `find_assort_multipliers` | Appendix A/B: H(h,x,i) | Same normalization function as in the replicator; applied to the RL signal distribution. |
| `executilcomputeREforget` | Appendix B: reinforcement + forgetting | (1) Applies forgetting: all urn entries × `forgetf`. (2) For all agent pairs, computes payoffs from `agentprofilepicks`, applies assortment multiplier, adds payoff to used urns only. (3) Enforces urn floor ≥ 1. |
| `execrepconvertRE` | Implementation detail | Extracts modal strategy from urn argmax for reporting. Populates `execurnsf` at reporting time only. |
| `genBS_resultsRE` | Sec 4.2 optimality criteria | Computes `simple_stat_001` / `simple_stat_002` from urn argmax values. |
| `time_update` | Table 4 | Same recording function as in the replicator. |

> **[➕ ADDED — No active mutation in RL loop]** `mutateratef` is passed to
> `genBS_f7_RothErevExec_full_play` but unused. No mutation functions are called in the
> active RL loop. Exploration comes from stochastic urn sampling and the floor constraint.

> **[➕ ADDED — Inactive functions in the RL script]** `greetings_check_success`,
> `genBS_check_success`, `social_draws`, `pairings`, `receiver_draws`, `connections_init`,
> `greetings_single_tstep`, `genBS_single_tstep`, `genBS_full_play`,
> `executilcomputeRE` (without forgetting) are present in the RL script but not called
> by the active RL notebook.

---

### What the Paper Left Implicit and How the Code Fills the Gap (Corrected)

| Paper phrase | Implicit detail | Code implementation |
|:---|:---|:---|
| "cascade rounding fixed the statistical anomaly" | Exact method not described | `flip7zero_correction` reinitializes a type's profiles randomly when cascade rounding yields zero agents for that type |
| "forgetting factor 0 < φ < 1" | No concrete value or update ordering | In `executilcomputeREforget`: forgetting (× φ) is applied **first**, then reinforcement is added, then floor ≥ 1 is enforced |
| "assortment multipliers based on base_connection_weights" | Exact normalization omitted | `execassort_array_create` builds raw weights; `find_assort_multipliers` normalizes by population signal counts each timestep |
| "mutations applied every 10 steps" | Not in paper | Main loop checks `if idxN0 % 10 == 0` before calling `muterandoms10_smart` (replicator only; RL has no mutations) |
| "null signal represents 'do not attend'" | Masking rule not spelled out | In `executilcompute` / `executilcomputeREforget`: any dimension where either agent has a 0 signal is zeroed for both when computing effective signals. Assortment uses pre-masking broadcast signals. |

---

## 3. Exhaustive Cross-Check (Corrected)

> **[⚠ CORRECTED — ❌ Multiple entries in the cross-check table are wrong]**
> Several entries in the prior document's cross-check table attribute functions to both
> notebooks that only belong to one, or attribute legacy functions as active, or describe
> `execurnsf` as dynamically updated. These are corrected below.

| Paper / Supplement Component | Corresponding Code | Active? | Correct? |
|:---|:---|:---|:---|
| Definition of a "type" (preference group) | `popt` (via `population_array`) | Both notebooks | ✓ |
| Alpha-signal selection (most attended dimension for type 0) | `getalphastyped0` / `getalphasignaling` | Both notebooks (post-processing) | ✓ |
| Executive attention mask | Null-signal masking logic in `executilcompute` / `executilcomputeREforget`; `execurnsf` populated at **reporting time only** by `execrepconvert`/`execrepconvertRE` | Both | ⚠ Prior doc incorrectly said `execurnsf` is "updated in `executilcomputeREforget`" |
| Signal cost (negative payoff per attended dimension) | `sigcostf` in `executilcompute` / `executilcomputeREforget` | Both | ✓ |
| Reinforcement / punishment parameters | `genBSpunishf` (= 0 in active sweeps); payoff via `coordination_preferencesf` | Both | ⚠ Prior doc cited `greetings_check_success` which is legacy |
| Replicator update rule | `genBSreplicator_single_tstep` | Replicator only | ✓ |
| Cascade rounding / zero-error correction | `flip7zero_correction` | Replicator only | ✓ |
| Utility calculation U_k(x) | `executilcompute` (active replicator) | Replicator only | ⚠ Prior doc also listed `utilcompute` and `executilcomputeRE` as active; neither is |
| Assortment multiplier H(h,x,i) | `execassort_array_create`, `find_assort_multipliers` | Both | ⚠ Prior doc listed `assort_array_create` (non-exec) as equivalent; exec version is what matters |
| Signal-count → assort-multiplier mapping | `get_sig_counts` (replicator) / `get_sig_countsRE` (RL) | Both | ✓ |
| Time-series recording | `time_update` | Both | ✓ |
| Final aggregation of simple-stat matrices | `genBSexec_results` (replicator) / `genBS_resultsRE` (RL) | Both | ✓ |
| Mutual information calculation | `mutual_info`, `average_mutual_info` | Both | ✓ |
| RL urn drawing | `random2picks` → `agentprofilepicks` | RL only | ⚠ Prior doc described `random2picks` as "converting updated urns back into profiles"; it draws profiles at timestep start |
| RL reinforcement + forgetting | `executilcomputeREforget` | RL only | ✓ in concept, but prior doc also described this as part of the replicator path |
| Mutation (m₀, m₁) | `muterandoms10_smart`, `repmutation_smart` | **Replicator only** | ⚠ Prior doc described mutation as active in RL notebook; it is not |
| Pairings / weighted random pairing | NOT active in either notebook | Legacy only | ⚠ Prior doc listed `pairings`, `connections_init` as active |

---

## 4. Summary – Mapping at a Glance (Corrected)

> **[⚠ CORRECTED — ❌ The prior document's summary table has notebooks A and B swapped,
> wrong function lists, and legacy functions listed as active]**
> The prior table labeled the BZforget3 notebook as "Replicator" and the FLIP7 notebook
> as "Roth-Erev." Both labels and both function lists are reversed. Corrected below.

| Notebook | Learning Dynamics | Core Components | Representative Active Functions |
|:---|:---|:---|:---|
| **BZforget3 (RL):** `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep...ipynb` | **Roth-Erev Reinforcement with Forgetting** (Appendix B) | • Urn initialization (in notebook) to `inertia=1` • Timestep profile drawing (`agentprofilepicks`) • Assortment weighting • Forgetting + reinforcement + floor • Modal reporting | `random2picks`, `executilcomputeREforget`, `get_sig_countsRE`, `find_assort_multipliers`, `execrepconvertRE`, `genBS_resultsRE` |
| **FLIP7 (Replicator):** `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep...ipynb` | **Discrete Replicator Dynamics** (Section 2.2) | • Random profile initialization • Utility computation • Replicator update + cascade rounding • Two-parameter mutation • Conversion back to urns | `genBSreplicatorexec_k_step`, `executilcompute`, `genBSreplicator_single_tstep`, `flip7zero_correction`, `muterandoms10_smart`, `repmutation_smart`, `execrepconvert`, `genBSexec_results` |

---

## A. Variable Walk-Through: Replicator Dynamics
### Script: `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`

> **[⚠ CORRECTED — ❌ Prior document header for this section named the RL script]**
> The prior document's Section A header read "Replicator-Dynamics (exec-based) –
> *genBS_f7_RothErev_rep_execNULLsig.py*". That is the RL script. The replicator script
> is `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`.

| Variable | Shape/Type | Paper Reference | What the Code Does |
|:---|:---|:---|:---|
| `popt` | 1D int array, length N | Sec 1.1: preference type of each individual | Agent-to-type mapping. Used as index in all type-partitioned aggregations. |
| `profile_count_typed` | 2D float array [numtypes, numprofiles] | Sec 2.2: N_{k,t}(x) | Tracks number of agents of type k using profile x. Core replicator state. |
| `profile_count_untyped` | 1D float array [numprofiles] | Sec 2.2: M(i) = Σ_k N_{k,t}(x) | Total agents per profile across all types. Used as M(i) in utility calculation. |
| `agentprofiles` | 1D int array, length N | Sec 2.2: agent's current strategy | Per-agent profile index. Rebuilt by `genBSreplicator_single_tstep` each timestep. **[⚠ CORRECTED: prior doc said initialized by `genBSreplicator_first_tstep` which is commented out; active initializer is `genBSreplicatorexec_k_step`]** |
| `profile_utility_bytype` | 2D float array [numtypes, numprofiles] | Sec 2.2: U_k(x) | Expected payoff matrix. Produced by `executilcompute`. Drives the replicator update. |
| `assort_multipliers` | 2D float array [numsignals, numsignals] | Appendix A: H(h,x,i) | Normalized assortment weights. Produced by `find_assort_multipliers` each timestep. |
| `assort_sig_counts` | 1D int array [numsignals] | Appendix A: denominator for H | Signal count vector. Produced by `get_sig_counts`. |
| `profiles_indexed` | 2D int array [numprofiles, sigdimensions+numsignals] | Sec 4.1: strategy profile strings | Lookup table mapping integer index to full strategy vector. |
| `sigurnsf` | List of 2D arrays, each [population, signals_per_dim] | Sec 3.1: signal urns | In replicator, initialized as random one-hot vectors. Rebuilt from profiles at reporting time by `execrepconvert`. **[⚠ CORRECTED: prior doc said urns "start empty and are re-filled each timestep"; in the replicator they are one-hot vectors set at initialization and rebuilt for reporting only]** |
| `recurnsf` | 3D array [population, numsignals, numactions] | Sec 3.1: receiver urns | Same: initialized as one-hot, rebuilt for reporting. Updated by `greetings_check_success` is WRONG — that is a legacy function. |
| `execurnsf` | 2D int array [population, sigdimensions] | Sec 3.3: attention flags | **Reporting-time only.** Inferred from signal urn argmax by `execrepconvert`. Not updated during dynamics. **[⚠ CORRECTED: prior doc said "updated in `executilcomputeREforget`"; it is not]** |

---

## A (continued). What the Paper Left Implicit — Replicator Model

| Implicit detail | Code implementation |
|:---|:---|
| How zero-counts after rounding are handled | `flip7zero_correction`: reinitializes the type's profile distribution randomly |
| How mutation masks are generated efficiently | `muterandoms10_smart`: pre-generates 10 rounds of mutation draws; refreshed every 10 timesteps |
| How strategy profiles are decoded from integers | `profiles_indexed`: O(1) lookup table mapping each profile index to its full signal/action vector |
| How convergence is detected | Main loop compares `agentprofilesCOPY` to `agentprofiles` every 100 timesteps |

---

## B. Variable Walk-Through: Roth-Erev Reinforcement Learning
### Script: `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`

| Variable | Shape/Type | Paper Reference | What the Code Does |
|:---|:---|:---|:---|
| `sigurnsf` | List of 2D arrays, each [population, signals_per_dim] | Appendix B: sender urns | Initialized to `inertia=1` in notebook. Weights are updated by `executilcomputeREforget` (forgetting + selective reinforcement). |
| `recurnsf` | 3D array [population, numsignals, numactions] | Appendix B: receiver urns | Same initialization and update pattern as `sigurnsf`. |
| `agentprofilepicks` | 2D int array [population, profile_length] | Appendix B: "strategy profile for entire timestep" | Produced by `random2picks` at the start of each timestep. The full signal/action draw from urns. Consumed by `executilcomputeREforget`. Overwritten each timestep. **[➕ ADDED: not mentioned in prior document's variable sections]** |
| `forgetf` | scalar float (0.998) | Appendix B: φ forgetting factor | Applied to all urn entries at the start of each timestep's `executilcomputeREforget` call, **before** reinforcement. |
| `execurnsf` | 2D int array [population, sigdimensions] | Sec 3.3: attention flags | **Reporting-time only.** Inferred from signal urn argmax by `execrepconvertRE`. Not updated during RL dynamics. |

> **[⚠ CORRECTED — ❌ Prior document's Section 2.2 described the wrong RL loop]**
> The prior document described the RL notebook as calling:
> - `genBS_single_tstep` (via `social_draws` → `pairings` → `receiver_draws` → reinforcement)
> - `greetings_full_play_typed`, `genBSreplicatorexec_full_play_typed` as "wrappers"
> - `mutations10`, `muterandoms10_smart`, `repmutation_smart` as active
>
> **None of these descriptions are correct for the active RL notebook.** Specifically:
> - `social_draws`, `pairings`, `receiver_draws`, `greetings_check_success`, `genBS_check_success`,
>   and `genBS_single_tstep` are legacy functions **not called** by `genBS_f7_RothErevExec_full_play`.
> - `genBSreplicatorexec_full_play_typed` is the **replicator** main loop, not an RL wrapper.
> - `greetings_full_play_typed` and `genBS_full_play_typed` are legacy main loops, not type
>   converters for saving results.
> - Mutation functions are passed as parameters to the RL function but are not called within it.

> **[⚠ CORRECTED — ❌ `random2picks` described incorrectly]**
> The prior document (line 64) described `random2picks` as "the step that converts the
> updated urns back into integer strategy profiles for the next timestep" and as implementing
> "replace the 0 signal with the previous alpha signal value." **Both descriptions are wrong.**
> `random2picks` draws a fresh strategy profile from the current urn weights at the **start**
> of each timestep and stores it in `agentprofilepicks`. It does not convert updated urns at
> the end of a timestep, and it has no logic involving "alpha signals."

---

## B (continued). What the Paper Left Implicit — RL Model

| Implicit detail | Code implementation |
|:---|:---|
| Forgetting occurs before reinforcement | Within `executilcomputeREforget`, urn multiplication by φ happens first; payoff additions happen after |
| How all-pairs interaction is achieved | `executilcomputeREforget` loops over all ordered agent pairs; not random pair sampling |
| How urn floor is applied | After reinforcement loop: `if urn_entry < 1: urn_entry = 1` |
| How convergence is detected | Main loop compares `agentprofilepicksCOPY` to `agentprofilepicks` every **1,000** timesteps (vs. 100 for replicator, because stochastic draws from stable urns still differ on consecutive timesteps) |
| Why coordination preferences are scaled by 1/200 | Keeps reinforcement increments (≈0.005–0.0075 per interaction) small relative to initial urn weights of 1 |

---

## Final Take-Away (Corrected)

The two notebooks are complete implementations of the model:

- **BZforget3 (RL) notebook** implements Roth-Erev reinforcement learning with forgetting
  (Appendix B). Its core loop draws profiles from urns (`random2picks`), computes all-pairs
  payoffs with assortment weighting, reinforces used urns selectively, and applies forgetting
  and a floor constraint (`executilcomputeREforget`).

- **FLIP7 (Replicator) notebook** implements discrete replicator dynamics (Section 2.2).
  Its core loop computes population-level expected utilities (`executilcompute`), applies the
  replicator update equation with cascade rounding (`genBSreplicator_single_tstep`), and
  applies two-parameter mutation (`muterandoms10_smart` / `repmutation_smart`).

Both notebooks share assortment infrastructure (`execassort_array_create`,
`find_assort_multipliers`) and post-processing functions (`time_update`, `getalphasignaling`,
`mutual_info`). The only places where the code adds information not in the paper are small
technical details: cascade rounding, zero-count correction, epsilon-offset sampling, the
forgetting-before-reinforcement ordering, the all-pairs interaction structure, and the 1/200
scaling of coordination preferences in the RL notebook.
