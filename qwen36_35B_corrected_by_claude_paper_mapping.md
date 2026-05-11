# Code–Paper Correspondence: Social Identity Signaling Simulations
## Corrected Review of Second-Model Attempt (Qwen 35B)

> **About this document:** This document corrects and supplements the second model's mapping
> between the simulation code and the paper/supplement. All modifications are marked:
> - **[⚠ CORRECTED]** — a factual error has been fixed.
> - **[➕ ADDED]** — new material not present in the prior document.
> - **[❌ PRIOR ERROR]** — a brief inline label identifying the specific mistake being corrected.
>
> Unmarked text is carried over from the prior document and is not disputed.

---

### 🔍 IDENTIFICATION OF LEARNING DYNAMICS

> **[➕ ADDED — Filename note]** The notebook filenames in the identification table below include
> the suffix `_short` (e.g., `...-Copy34_short.ipynb`), which does not appear in the actual
> uploaded files (`...-Copy34.ipynb`). This appears to be an artifact of the model's context
> window. The identification of dynamics and scripts is otherwise correct.

| Notebook | Associated Script | Learning Dynamic Implemented | Key Identifier in Code/Notebook |
|---|---|---|---|
| `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb` | `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py` | **Roth-Erev Reinforcement Learning with Forgetting** (Supplement Appendix B) | Imports `genBS_f7_RothErevExec_full_play_typed`, sets `BZforget_multiplier = 0.998`, uses `executilcomputeREforget` for update step |
| `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb` | `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py` | **Discrete Replicator Dynamics** (Paper Section 2.2) | Imports `genBSreplicatorexec_full_play_typed`, uses `genBSreplicator_single_tstep` for update step, no forgetting factor |

> **[➕ ADDED]** This identification table is notably more accurate than the first model's attempt,
> which incorrectly stated that both notebooks drew from a single shared script and got one
> notebook's version number wrong. Here both filenames and both script names are correct.

---

### 📘 PART 1: DISCRETE REPLICATOR DYNAMICS

**Notebook:** `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`
**Script:** `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`

#### 1. Strategy Representation & Initialization

- **Paper Description (Sec 2.2 & 4.1):** Agents possess strategy profiles (strings of signals and contingent actions). Initial strategies are randomly assigned.
- **Code Implementation:**
  - `population_array`: Maps type distribution to individual agents.
  - `replicatorexec_initialize`: Draws one-hot random actions/signal dimensions for each agent's sender and receiver urns. Initializes `profiles_indexed`, a base-N indexing array that maps every valid strategy profile (signal dimensions × action contingencies) to a unique integer ID.
  - `repmultipliers` (in notebook): Computes the positional multipliers needed to convert multi-dimensional strings into single profile indices.

  **[➕ ADDED]** After `replicatorexec_initialize` sets up the structural arrays, the active
  random initialization of profile assignments uses **`genBSreplicatorexec_k_step`** (not
  `genBSreplicator_first_tstep`, which is commented out in the active code). This function
  assigns each agent a uniformly random profile index from all `numprofiles` possibilities,
  implementing the paper's "randomly assigned strategy profiles." `replicator_initialize`
  (without the "exec" prefix) is present in the script but used only in older variants of the
  replicator that lack the attention/cost mechanism; it is not called by the active notebook.

#### 2. Interaction & Assortment

- **Paper Description (Appendix A, Sec 3.4, 4.1):** Payoffs are weighted by homophily factor H(h,x,i).
- **Code Implementation:**

  > **[⚠ CORRECTED — ❌ `connections_init` and `pairings` are not active in the replicator]**
  > The prior document listed `connections_init` and `pairings` as part of the replicator
  > model's assortment mechanism. **Neither function is called by `genBSreplicatorexec_full_play`.**
  > They belong to older pair-based variants (`greetings_single_tstep`, `genBS_single_tstep`)
  > that are present in both scripts but not invoked by either active notebook. The active
  > replicator computes assortment analytically, not via explicit pairing.

  - `execassort_array_create`: Pre-computes the full `[numsignals × numsignals]` static homophily matrix. **Critically, it only counts signal matches in dimensions where both agents have non-null signals** — dimensions where either agent broadcasts null (0) are excluded from the match count. This is distinct from `assort_array_create` (without "exec"), which counts all matching dimensions including mutual-null ones; that function belongs to older variants and is not active in this notebook.
  - `find_assort_multipliers`: Normalizes the static homophily matrix against current population signal frequencies to produce `assort_multipliers[i][j]` = H(h,x,i) for every signal pair. This implements the full Appendix A formula: H(h,x,i) = N / Σ_j[S(h,x,j)] × S(h,x,i).
  - `get_sig_counts`: Tallies how many agents currently broadcast each combined signal index; provides the Σ_j denominator for normalization.

#### 3. Utility/Payoff Calculation

- **Paper Description (Sec 2.2):** U_k(x) = Σ_{i∈Y} M(i) × p_{xi,k}, where M(i) is the count of agents playing profile i, and p_{xi,k} is the pairwise payoff. With assortment: weighted by H(h,x,i) (Appendix A).
- **Code Implementation:**
  - `executilcompute`: **The active utility function.** Iterates over all profile pairs present. For each pair (x, i), calculates the effective signal each agent sees (zeroing any dimension where either agent has a null signal), retrieves contingent actions, checks coordination success/failure, and adds `coordination_preferencesf` or `genBSpunishf` weighted by `profile_count_untyped[i]` (M(i)) and `assort_multipliers`. Adds signal costs (`totalAsigcost`) for each active signaling dimension.

  > **[⚠ CORRECTED — ❌ `utilcompute` is not active in the replicator notebook]**
  > The prior document listed `utilcompute` and `executilcompute` together as implementing the
  > utility calculation. In `genBSreplicatorexec_full_play`, only `executilcompute` is called.
  > `utilcompute` (without the "exec" prefix) lacks signal cost handling and is used only in
  > older replicator variants (`genBSreplicator_full_play`). It is not invoked by the active
  > replicator notebook.

#### 4. Replicator Update Rule

- **Paper Description (Sec 2.2):** N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) × [U_k(x) - Avg(U_k)]
- **Code Implementation:**
  - `genBSreplicator_single_tstep`: Implements the discrete replicator equation.
    - Computes `typesaverage` by averaging utilities across present profiles for each type.
    - Updates counts: `profile_count_typed += profile_count_typed * (utility - avg_utility)`
    - Enforces non-negativity (< 0 → 0).
    - Normalizes and scales by type population size.
    - **Cascade Rounding:** Uses cumulative floating-point accumulation (`floattotal`) combined with `np.around` to convert floats back to integer agent counts while preserving totals. This computational detail is not in the paper but is necessary for discrete agent-based simulation.

**[➕ ADDED]** `flip7zero_correction`: A numerical safety function called within
`genBSreplicator_single_tstep` when the replicator dynamics collapse all agents of a given type
into profiles with utilities so far below average that rounding produces a zero total for that
type. It randomly reinitializes the type's distribution. The code notes this is a ~1 in 10,000
chance event. Not described in the paper.

#### 5. Mutation

- **Paper Description (Sec 2.2, lines 207–210):** Mutation governed by m₀ (selection probability) and m₁ (element-wise flip probability).
- **Code Implementation:**
  - `muterandoms10_smart`: Generates random masks based on m₀ and proposes new profile elements based on m₁. Pre-generates 10 rounds of mutations at once for efficiency; refreshed every 10 timesteps.
  - `repmutation_smart`: Applies masks to current profiles — each profile position uses the randomly drawn value with probability m₁, or retains the old value otherwise. Converts new profile strings to indices and updates `profile_count_typed` and `profile_count_untyped`.

  > **[➕ ADDED — Unused mutation variants]** `muterandoms10` and `repmutation` (without
  > `_smart`) are also present in the script but are **not** called by the active replicator.
  > The `_smart` variants are the active ones because they implement the two-parameter (m₀, m₁)
  > mutation described in the paper, whereas the non-smart versions use only m₀ and replace
  > entire profiles rather than individual elements.

**[➕ ADDED — Convergence check]** The active replicator loop checks for early termination every
100 timesteps by comparing `agentprofilesCOPY` to `agentprofiles`. If they are identical (the
distribution has fully stabilized), the simulation halts before reaching `runlength`. This
corresponds to the paper's footnote: "every 100 timesteps it was checked whether the distribution
of agents' strategy profiles was unchanged."

#### 6. Recording & Output

- **Code Implementation:**

  > **[⚠ CORRECTED — ❌ `genBSreplicatorexec_k_step` is an initialization function, not part of recording/output]**
  > The prior document listed "`genBSreplicatorexec_k_step` / `genBSreplicatorexec_full_play`:
  > Main simulation loop." `genBSreplicatorexec_k_step` is the **random profile initialization**
  > function called once at the start of the simulation; it has nothing to do with recording or
  > output. `genBSreplicatorexec_full_play` is the main simulation loop. These serve completely
  > different roles.

  - `genBSreplicatorexec_full_play`: Main simulation loop.
  - `execrepconvert`: Converts integer profile counts back to one-hot urn structures for reporting.
  - `genBSexec_results`: Aggregates behavioral statistics (social signal frequencies, action contingencies, `simple_stat_001`, `simple_stat_002`) from the urn format.
  - `time_update`: Records time-series of conditional action probabilities.
  - `average_mutual_info` / `getalphasignaling`: Computes Shannon mutual information and identifies "alpha" signals for cross-run aggregation.

---

### 📗 PART 2: ROTH-EREV REINFORCEMENT LEARNING WITH FORGETTING

**Notebook:** `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`
**Script:** `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`

#### 1. Urn Initialization

- **Supplement Description (App B):** Each agent maintains an urn for each dimension/action in their strategy string. Each urn starts with 1 ball per possible action.

> **[⚠ CORRECTED — ❌ `RothErevExec_initialize` does not initialize urn entries]**
> The prior document states: "`RothErevExec_initialize`: Sets up `sigurnsf` and `recurnsf`
> arrays. Initializes all entries to `inertia=1`." **This is wrong.** `RothErevExec_initialize`
> only computes the structural parameters `sigperdimf` and `profilecaps` (the number of signals
> per dimension and the size of each urn position). It returns only these two arrays and does
> not touch `sigurnsf` or `recurnsf` at all. The urn entries are initialized to `inertia=1`
> in the **notebook itself**, before the simulation call, via:
> `sigurns = [np.ones([population_size, numsignals_perdim[i]]) * inertia for i in ...]`
> `recurns = np.ones([population_size, numsignals, numactions]) * inertia`
> This corresponds to Appendix B's "one ball corresponding to each possible action."

#### 2. Timestep Strategy Selection (Computational Tweak)

- **Supplement Description (App B):** "We found that having agents select a single strategy profile that they employ for all interactions on a single timestep made this learning dynamic significantly more computationally tractable."
- **Code Implementation:**
  - `random2picks`: Draws one discrete action/signal per dimension from each agent's urns for the current timestep. Stores the result in `agentprofilepicks`. This single profile is held fixed for all pairwise interactions during that timestep, exactly matching the supplement's tractability optimization.

#### 3. Pairing & Assortment

- **Supplement Description (App B):** Payoffs are multiplied by H(h,x,i) based on shared signal dimensions.

> **[⚠ CORRECTED — ❌ `connections_init` and `pairings` are not active in the RL model]**
> The prior document states the RL notebook "Uses the same assortment infrastructure as the
> replicator model (`connections_init`, `pairings`, `assort_array_create`, `find_assort_multipliers`)."
> **`connections_init` and `pairings` are NOT called by `genBS_f7_RothErevExec_full_play`.**
> Like the active replicator, the active RL model uses the analytical assortment approach
> (`execassort_array_create` and `find_assort_multipliers`), not explicit per-agent pairing.
> `connections_init` and `pairings` are legacy functions from earlier pair-based RL variants
> that are not invoked by either active notebook.
>
> Additionally, the prior document mentioned `assort_array_create` (without "exec") as active.
> The active RL model calls `execassort_array_create`, which only counts matches in dimensions
> where both agents have non-null signals, matching the null-signal attention model.

#### 4. Payoff Calculation & Reinforcement

- **Supplement Description (App B):** If payoff > 0, reinforce the specific urns that contributed to the action. Add payoff-many balls to the used urns.
- **Code Implementation:**
  - `executilcomputeREforget`: Core reinforcement loop.
    - Iterates through all agent pairs using the fixed timestep profiles from `agentprofilepicks`.
    - Computes observed signals (masking unattended dimensions) and retrieves contingent actions.
    - Calculates payoff using `coordination_preferencesf`, `sigcostf`, and `assort_multipliers`.
    - **Selective Reinforcement:** Adds the calculated payoff (`Areinforcement`/`Breinforcement`) **only** to the specific sender dimension urn entries and receiver urn entries that were actually used in that interaction. Uninvolved urns are untouched. Matches the supplement's urn update rule precisely.

#### 5. Forgetting Mechanism

- **Supplement Description (App B):** At timestep end, multiply all urn contents by forgetting factor φ, and floor any urn entry below 1 to 1.
- **Code Implementation:**
  - `executilcomputeREforget`: At the **very beginning** of the function (before reinforcement), multiplies all `sigurnsf` and `recurnsf` entries by `forgetf` (set to 0.998 in notebook). After reinforcement, enforces `if urn < 1: urn = 1`. This implements the forgetting floor exactly as described.

> **[⚠ CORRECTED — ❌ Description of `executilcomputeREforget` incorrectly states it calls `random2picks` internally]**
> In Table IV.B, the prior document describes `executilcomputeREforget` as: "2. Draws one
> profile via `random2picks`." **`executilcomputeREforget` does not call `random2picks`.**
> The profile draws are performed by `random2picks` in the **main loop** of
> `genBS_f7_RothErevExec_full_play`, before `executilcomputeREforget` is called. The completed
> `agentprofilepicks` array is then passed as a parameter into `executilcomputeREforget`. The
> two functions are sequential steps in the loop, not nested. The correct order in the main loop
> is: `random2picks` → `get_sig_countsRE` → `find_assort_multipliers` → `executilcomputeREforget`.

**[➕ ADDED — No active mutation in RL]** The `mutateratef` parameter is passed to
`genBS_f7_RothErevExec_full_play` but is **not used** inside that function. There are no mutation
calls in the active RL loop. Exploration is provided instead by: (1) stochastic urn sampling via
`random2picks` each timestep, and (2) the floor constraint (minimum urn value = 1) that keeps
every option permanently available regardless of how little it has been reinforced.

**[➕ ADDED — Convergence check]** The active RL loop checks for early termination every 1,000
timesteps (vs. 100 for the replicator), comparing `agentprofilepicksCOPY` to `agentprofilepicks`.
The longer interval reflects that stochastic urn sampling produces different draws on successive
timesteps even at a stable equilibrium, making exact equality less useful at short intervals.

#### 6. Recording & Output

- **Code Implementation:**
  - `genBS_f7_RothErevExec_full_play`: Main simulation loop. Replaces the replicator step with `random2picks` → `executilcomputeREforget`.
  - `execrepconvertRE` / `genBS_resultsRE`: Extracts behavioral statistics from the continuous urn values (using `np.argmax` to determine current behavioral mode) and aggregates them for analysis.
  - `time_update`, `average_mutual_info`, `getalphasignaling`: Same analysis functions as the replicator, applied to the RL trajectory.

---

### ✅ COMPLETENESS VERIFICATION

> **[⚠ CORRECTED — ❌ Table lists `utilcompute` and `replicator_initialize` as active in the replicator]**
> Two entries in this table require correction.

| Model Component (Paper/Supplement) | Connected Code Subfunction(s) | Notes |
|---|---|---|
| Discrete replicator equation | `genBSreplicator_single_tstep` | Exact algebraic translation + cascade rounding |
| Utility calculation | `executilcompute` (**not `utilcompute`**, which is inactive) | Handles M(i), coordination preferences, signal costs, assortment |
| Mutation (m₀, m₁) | `repmutation_smart`, `muterandoms10_smart` | Bitmask sampling & profile re-indexing; `_smart` versions are active, non-`_smart` versions are not |
| Random initial strategies | `replicatorexec_initialize` + `genBSreplicatorexec_k_step` | `replicatorexec_initialize` sets up structural arrays; `genBSreplicatorexec_k_step` assigns random profile indices. `replicator_initialize` (without "exec") is not active in this notebook |
| Multidimensional signals & executives | `executilcompute`, `execrepconvert`, `execassort_array_create` | Dimensions masked when exec=0 (null signal in that dim); costs applied per attended dimension |
| Assortment/Homophily (H function) | `execassort_array_create`, `find_assort_multipliers` | Implements (2^d)^h via base_connection_weights array; `connections_init` is NOT active in either notebook |
| Urn-based learning (App B) | `RothErevExec_initialize` (structural setup only) + urn pre-init in notebook, `executilcomputeREforget` | Direct urn weight manipulation |
| Forgetting & floor(1) | `executilcomputeREforget` (first lines) | Explicit multiplication by `forgetf` and thresholding |
| Single-profile-per-timestep tweak (App B) | `random2picks` + loop structure in `genBS_f7_RothErevExec_full_play` | Profile fixed for entire timestep, then passed to `executilcomputeREforget` |
| Payoff reinforcement to used urns only | `executilcomputeREforget` inner loop | Only `sigurnsf[dim][agent][draw]` and `recurnsf[agent][signal][action]` updated |
| Time-series & mutual information tracking | `time_update`, `average_mutual_info`, `getalphasignaling` | Called in both notebooks for result aggregation |

> **[⚠ CORRECTED — ❌ "No orphaned functions" claim is false]**
> The prior document's Final Verification states: "No orphaned functions: All `@numba.jit`
> functions are... All are explicitly invoked in their respective notebooks." **This is
> demonstrably incorrect.** The following functions are present in one or both scripts but
> are **not called** by either active notebook's primary simulation path:
> `connections_init`, `pairings`, `social_draws`, `receiver_draws`,
> `greetings_check_success`, `genBS_check_success`, `greetings_single_tstep`,
> `genBS_single_tstep`, `greetings_full_play`, `genBS_full_play`,
> `genBSreplicator_full_play`, `genBSreplicatorexec_first_tstep`,
> `genBSreplicator_first_tstep`, `executilcomputeRE` (without forgetting),
> `utilcompute`, `replicator_initialize`, `greetings_results`.
> These represent earlier model versions and are present as legacy code. Acknowledging them
> is important because mistakenly attributing them to the active simulation path — as both
> prior documents did in various places — leads to systematic misidentification of the active
> model components.

---

### 📝 SUMMARY

> **[⚠ CORRECTED — ❌ Summary lists `utilcompute` as active]**
> The prior summary states Notebook 2's core logic flows through "`utilcompute`/`executilcompute`
> (payoff)". Only `executilcompute` is active; `utilcompute` is not called by the replicator
> notebook.

- **Notebook 1** implements **Roth-Erev Reinforcement Learning with Forgetting** (Supplement Appendix B). Its core logic flows through `random2picks` → `get_sig_countsRE` → `find_assort_multipliers` → `executilcomputeREforget` (forgetting then reinforcement) → statistical aggregation.
- **Notebook 2** implements **Discrete Replicator Dynamics** (Paper Section 2.2). Its core logic flows through `get_sig_counts` → `find_assort_multipliers` → `executilcompute` (payoff) → `genBSreplicator_single_tstep` (replicator update + cascade rounding) → `repmutation_smart` (mutation) → statistical aggregation.

Both implementations faithfully translate the paper's and supplement's theoretical descriptions
into executable, high-performance (numba-compiled) agent-based simulations, with the code
explicitly filling in necessary computational details (indexing schemes, rounding algorithms,
urn flooring, and the single-profile timestep optimization) that are implicit or omitted in the
written text.

---

## 📘 I. SIMULATION & SWEEP PARAMETERS (Notebook-Level)

These variables control experimental design, population structure, and parameter sweeps.

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `runs` | 100 (NB1) / 1000 (NB2) | Paper Table 4: Number of simulations | NB1 matches Supplement App B (100 runs for RL). NB2 matches Table 4 (1000 runs for Replicator). |
| `runlength` | 4×10^4 | Table 4: Simulation length T | Max timesteps; early termination applies (see convergence check notes above). |
| `record_interval` | runlength | Table 4 | Records terminal state only. |
| `signal_snapshots` | 1 | Table 4 | Only final behavioral snapshot archived per run. |
| `population_size` | 500 (NB1) / 1000 (NB2) | Supplement App B: N=500 | NB1 matches supplement exactly. NB2 matches Table 4 replicator sweeps. |
| `numtypes` | 3 | Sec 3.3/4.2: Lagash/Girsu/Akk | Fixed throughout; types never change, only strategies do. |
| `mutation_rate` | np.array([0.01, 0.1]) | Sec 2.2: m₀=0.01, m₁=0.1 | m₀: probability an agent is selected for mutation. m₁: probability each element in their strategy string flips. **Used only in the replicator; passed to but unused by the RL function.** |
| `sigdimensions` | 2 | Sec 4.1: K=2 signaling dimensions | Two independent identity axes. |
| `numsignals_perdim` | np.array([2, 2]) | Sec 4.1: "one signal plus the null signal" | Per-dimension signal count. Total signal combinations = 2×2 = 4. |
| `base_connection_weights` | np.array([1, 2, 4, 8, 16, 32]) | Supplement App B: H(h,x,i) = (2×d)^h | `base_connection_weights[d] = 2^d`. When raised to `homophily_factor`, yields (2^d)^h. |
| `homophily_factor` | 0 (NB1) / 1 (NB2) | Supplement App B: h parameter | 0 = random mixing (H=1 for all pairs). 1 = full assortment. |
| `signal_cost` | -0.000125 (NB1) / -0.0005 (NB2) | Sec 3.3/4.1: cost c per dimension | Negative value added to utility. Applied per active (non-null) signaling dimension. |
| `genBSpunish` | np.ones((2))*0 | Sec 2.2: failure payoff = 0 | Coordination failure payoff = 0 in these sweeps. Array shape (2) is a legacy placeholder for potential asymmetric punishment. |
| `coordination_preferences` | [[1.5,0,.85],...]/200 (NB1) / [[1,0,.5],...] (NB2) | Table 7: Preference matrices | NB1 scaled by /200 to keep reinforcement increments tractable relative to initial urn weights of 1. NB2 uses raw scale matching replicator utility magnitudes. |
| `payoff_alphas`, `cents_offset` | Sweep over α, β | Sec 4.2: α preference asymmetry, β type proportion | α adjusts preferred payoff (1+α). cents_offset adjusts type proportions (0.33±β). |
| `BZforget_multiplier` | 0.998 | Supplement App B: φ=0.998 | Forgetting factor for Roth-Erev urns. |
| `inertia` | 1 | Supplement App B: "one ball corresponding to each possible action" | Sets initial urn weight in the notebook. Ensures every action has non-zero probability from the start. |
| `epsilon` | float_info.epsilon | N/A | Numerical stability. Ensures sampling from (0,1] instead of [0,1). |

---

## 📗 II. CORE MODEL STATE & STRATEGY REPRESENTATION

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `sigurnsf` | List of arrays, each [pop, sig_per_dim] × inertia | Supplement App B: Sender urns | One list element per dimension. `sigurnsf[d][agent][signal_index]` = weight for agent broadcasting signal in dimension d. |
| `recurnsf` | np.ones([pop, num_signals, num_actions]) × inertia | Supplement App B: Receiver urns | 3D array. `recurnsf[agent][observed_signal][action_index]` = weight for agent taking action given combined signal. |
| `execurnsf` | np.zeros([pop, sigdimensions]) | Sec 3.3/4.1: Executive attention | Not updated during dynamics. Inferred at reporting time from signal urn argmax: 0 if modal signal is null, 1 otherwise. |
| `profiles_indexed` | Generated by `replicator_initialize` / `replicatorexec_initialize` | Sec 4.1: Strategy profile strings | Base-N indexing table mapping every valid multi-dimensional strategy to a unique integer. Enables O(1) lookups. |
| `repmultipliers` / `sigmultipliers` | np.cumprod + np.roll | Sec 4.1: String length K+θ | Positional weight arrays. `sigmultipliers` handles signal dimensions only (length = sigdimensions = 2, values = [1,2]). `repmultipliers` handles signals + actions (longer). |
| `poptf` | `population_array(...)` | Sec 2.2: Fixed type distribution | 1D array: agent_index → type_index. Types never change; only strategies evolve. |
| `agentprofilepicks` | np.zeros([pop, profile_length], dtype=uint16) | Supplement App B: Single profile per timestep | RL only. Stores the strategy profile drawn from urns for the current timestep. Held fixed across all pairwise interactions in that timestep. |

---

## 📙 III. ASSORTMENT, PAIRING & INTERACTION MECHANICS

> **[⚠ CORRECTED — ❌ `conweightsf`, `pairsf`, `connections_init`, `pairings` are not active]**
> The prior table presented `conweightsf`, `pairsf`, `connections_init`, and `pairings` as
> active components of the assortment and pairing mechanism. **None of these is called by either
> active notebook's main simulation function.** They belong exclusively to legacy pair-based
> variants. The active mechanism uses `execassort_array_create` and `find_assort_multipliers`
> to compute assortment analytically at the population level, without explicit per-agent pairing.
> The corrected table below replaces `connections_init`/`pairings` with a legacy note.

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `assort_array` | `execassort_array_create(...)` | Supplement App A: Assortment matrix | Static [numsignals × numsignals] table. Entry [i][j] = `base_connection_weights[d]^h` where d = number of shared **non-null** signal dimensions. Built once before the main loop. |
| `assort_multipliers` | `find_assort_multipliers(...)` | Supplement App A: H(h,x,i) normalization | Normalized version of `assort_array`. `assort_multipliers[i][j] = assort_array[i][j] × (N / Σ_k assort_array[i][k] × count_k)`. Implements the full Appendix A normalization. |
| `assort_sig_counts` | `get_sig_counts` (replicator) / `get_sig_countsRE` (RL) | Supplement App A: Population signal census | Counts agents per signal combination each timestep; provides the Σ_j denominator for normalization. |
| `assortsigA` / `assortsigB` | Inside utility loops | Supplement App A: Agent's broadcast signal | Signal index based on actual (pre-masking) broadcast; used for assortment weighting. |
| `Asignal` / `Bsignal` | Inside utility/RL loops | Sec 4.1: Observed partner signal | Computed after masking unattended dimensions; used to look up contingent action from strategy profile. |
| `Aaction` / `Baction` | Inside utility/RL loops | Sec 4.1: Contingent action | Retrieved from strategy profile using the masked observed signal index. |
| `connections_init`, `pairings` | Legacy only | Legacy (not active) | Part of older pair-sampling variants. Not called by either active notebook. |

---

## 📕 IV. LEARNING DYNAMICS & UPDATE VARIABLES

### 🔹 A. Discrete Replicator Dynamics (NB2 / `genBSreplicatorexec_full_play`)

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `profile_count_typed` | np.zeros([numtypes, numprofiles]) | Sec 2.2: N_{k,t}(x) | Float array tracking number of agents of type k using profile x. Updated each timestep via replicator equation. |
| `profile_count_untyped` | np.zeros(numprofiles) | Sec 2.2: Σ_k N_{k,t}(x) | Aggregated frequency across all types. Acts as M(i) in utility calculation. |
| `typesaverage` | np.sum(...)/typeaveragecount | Sec 2.2: Avg(U_k(i)) | Average utility across all present profiles for a given type. Drives selection gradient. |
| `profile_utility_bytype` | `executilcompute` | Sec 2.2: U_k(x) | Expected payoff matrix. Sum over M(i) × p_{xi,k} weighted by assortment. Includes signal costs per active dimension. |
| `genBSreplicator_single_tstep` | Main update function | Sec 2.2: Discrete replicator equation | Implements N_{t+1} = N_t + N_t(U − Ū). Enforces non-negativity, renormalizes, then applies cascade rounding to convert floats to integer agent counts. |

### 🔹 B. Roth-Erev Reinforcement Learning (NB1 / `genBS_f7_RothErevExec_full_play`)

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `forgetf` | 0.998 | Supplement App B: φ | Forgetting factor. Applied to **all** urn contents at the **start** of each timestep's update (before reinforcement). |
| `Areinforcement` / `Breinforcement` | `executilcomputeREforget` | Supplement App B: "add payoff-many balls" | `assort_mult × (coord_payoff + sigcost)`. Added only to urns that contributed to the interaction. |
| `executilcomputeREforget` | Core RL update | Supplement App B: Reinforcement + Forgetting | 1. Multiplies all urns by `forgetf`. 2. Receives `agentprofilepicks` (profiles already drawn by `random2picks` before this function). 3. Interacts each agent with all others using that fixed profile. 4. Reinforces only used urns by calculated payoff. 5. Enforces urn ≥ 1. |
| `random2picks` | Timestep profile generation | Supplement App B: "select single strategy profile" | Draws one signal per dimension and one action per signal contingency simultaneously, forming a complete strategy profile held constant for the entire timestep. Called **before** `executilcomputeREforget`, not inside it. |

---

## 📕 V. MUTATION & NUMERICAL STABILITY VARIABLES

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `mutations10` | `numbagreater(...)` | Sec 2.2: m₀ selection | Boolean bitmask: 1 if agent selected for mutation, 0 otherwise. Generated via uniform random comparison against m₀. **Active only in replicator.** |
| `newprofiles10` / `muteprofilestrings` | `muterandoms10_smart` | Sec 2.2: m₁ element flip | Proposed mutation strings. Each element flips with probability m₁. `muteprofilestrings` uses bitmasking for vectorized element-wise mutation. **Active only in replicator.** |
| `repmutation_smart` | Update step | Sec 2.2: Mutation application | Applies masks to current profiles, converts new strings to flat indices via `repmultipliers`, updates `profile_count_typed`/`untyped`. Handles boundary conditions. **Active only in replicator.** |
| `flip7zero_correction` | `genBSreplicator_single_tstep` | N/A | **Computational fix**: If cascade rounding drives a type's total profile count to zero, redistributes agents randomly. Ensures simulation robustness. Not in paper. |
| `epsilon` in sampling | `rng.random()+epsilon` | N/A | **Computational fix**: Prevents cumsum sampling from ever returning exactly index 0 due to floating-point edge cases. Ensures ergodic sampling. |

---

## 📘 VI. RECORDING, OUTPUT & ANALYSIS VARIABLES

| Variable | Code Context | Paper/Supplement Mapping | Computational/Implementation Detail |
|---|---|---|---|
| `typed_time` / `signal_time` | `time_update` | Table 4: Time-series recording | 4D arrays: [type/signal][observed_signal][action][time_index]. Records conditional action probabilities over time. Normalized per agent to avoid popularity bias. Used for single-run trajectory visualization, not cross-run sweep statistics. |
| `simple_stat_001` / `simple_stat_002` | `genBS_results` / `genBSexec_results` / `genBS_resultsRE` | Sec 4.2: Optimal group structure criteria | `simple_stat_001`: modal action via argmax. `simple_stat_002`: stricter — requires >75% signal consistency AND >75% action consistency. `simple_stat_002` is the primary measure used for sweep heatmap classification. |
| `socialsig_aggregate` | `genBS_results` | Sec 3.4/4.2: Dominant signals | Counts how often each type broadcasts each multi-dimensional combined signal index. |
| `action_aggregate` | `genBS_results` | Sec 3.2/4.2: Conditional actions | Counts how often each type takes each action given each observed signal. |
| `multidimsignalscount0` / `runsmultidimsignalscount0` | `getalphasignaling` | Sec 4.2: Signal aggregation | Tracks the most common signal per type across runs. Because signal labeling is arbitrary, `getalphasignaling` relabels signals so that type 0's (Lagashites') most common signal is consistently labeled "alpha" — enabling meaningful cross-run averaging of signal distributions. |
| `execsruns` / `meanexecsruns` | `getalphasignaling` | Sec 3.3/4.1: Executive attention | Tracks proportion of agents in each type that attend to each signaling dimension. >0.5 threshold defines attending behavior. |
| `average_mutuinfo` | `average_mutual_info` | Sec 4.3: Information transmission | Computes Shannon mutual information I(S;A) between signals and actions. Used for the synergistic information analysis related to intersectional identity (Varley and Kaminski, 2022). |

---

## 🔍 VII. CRITICAL COMPUTATIONAL BRIDGING DETAILS (Code vs. Paper)

The paper presents abstract mathematical dynamics; the code implements them using discrete array operations. Here's how the bridge is constructed:

1. **String Notation → Flat Indexing**:
   - Paper: `<1BS>`, `<2AB>`, etc.
   - Code: `repmultipliers` + `profiles_indexed` map multi-dimensional coordinates to integers. Enables O(1) array lookups and vectorized operations instead of string parsing.

2. **Continuous Utility → Discrete Agent Counts**:
   - Paper: N_{k,t+1}(x) implies continuous frequencies.
   - Code: `genBSreplicator_single_tstep` uses **cascade rounding** (cumulative floating-point accumulation + `np.around`) to convert float utilities back to integer agent counts while preserving population totals.

3. **Urn Reinforcement → Selective Array Updates**:
   - Paper/Supplement: "Add payoff-many balls to the urn that it was drawn from."
   - Code: `executilcomputeREforget` uses `agentprofilepicks` to identify which `(dimension, signal, action)` indices were drawn, then updates **only** `sigurnsf[dim][agent][draw]` and `recurnsf[agent][signal][action]`. Uninvolved entries remain untouched, exactly matching supplement description.

4. **Assortment Formula → Matrix Precomputation**:
   - Paper/Supplement: H(h,x,i) = N / Σ_j[S(h,x,j)] × S(h,x,i)
   - Code: `base_connection_weights = [1,2,4,8...]` stores 2^d. `execassort_array_create` computes (2^d)^h and stores in a static matrix. `find_assort_multipliers` applies population-frequency normalization each timestep to produce the full H(h,x,i).

5. **Forgetting Floor → Explicit Thresholding**:
   - Paper/Supplement: "quantity raised to 1"
   - Code: `if sigurnsf[...] < 1: sigurnsf[...] = 1` after forgetting multiplication. Prevents vanishing probabilities and maintains exploration.

6. **Single-Profile-Per-Timestep Tweak**:
   - Supplement explicitly notes this tractability improvement. Code implements it via `random2picks` drawing once per timestep, storing in `agentprofilepicks`, then passed to `executilcomputeREforget` which iterates all pairs using that fixed profile.

7. **Signal Cost as Negative Addition**:
   - Paper: "payoff decreased by c"
   - Code: `sigcostf = -0.000125`. Utility calculation: `coord_payoff + totalAsigcost`. Since `totalAsigcost` is negative, it subtracts from payoff. Applied per non-null active dimension.

8. **[➕ ADDED] Null-Signal Attention Masking**:
   - Paper/Supplement (Sec 3.3): "When an agent does not attend to signals, she interacts with all other agents as if they had also broadcast 0."
   - Code: In `executilcompute` and `executilcomputeREforget`, before computing effective signals, any dimension where either agent has a null (0) signal is zeroed for both agents. Assortment multipliers, by contrast, use the actual (pre-masking) broadcast signals, because homophily depends on what is physically broadcast.

---

## ✅ FINAL VERIFICATION (Corrected)

- **Every major active variable** in both scripts has been accounted for.
- **Every paper/supplement equation** (replicator dynamics, utility calculation, reinforcement rule, forgetting, assortment, mutation, signal costs, attention masking) has a direct code equivalent in the **active** simulation paths.
- **Every computational implementation detail** (base-N indexing, cascade rounding, urn flooring, epsilon sampling, single-profile timestep, matrix precomputation, zero-correction) is mapped to its theoretical counterpart or justified as a necessary simulation bridge.
- **Legacy/inactive functions**: Several functions present in both scripts are **not** called by either active notebook (`connections_init`, `pairings`, `social_draws`, `receiver_draws`, `greetings_check_success`, `genBS_check_success`, `executilcomputeRE`, `utilcompute`, `replicator_initialize`, and others). These represent earlier model versions.
- **Consistency**: The two notebooks implement exactly the dynamics described in the paper (replicator) and supplement (Roth-Erev forgetting), using identical structural assumptions (multidimensional signals, executives, assortment) but different update rules and different Python scripts.
