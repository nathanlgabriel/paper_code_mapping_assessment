# Code–Paper Correspondence: Social Identity Signaling Simulations
## Corrected Review of Fourth-Model Attempt (Gemma4 26B)

> **About this document:** This document corrects and supplements the fourth model's mapping.
> All modifications are marked:
> - **[⚠ CORRECTED]** — a factual error has been fixed.
> - **[➕ ADDED]** — new material not present in the prior document.
> - **[❌ PRIOR ERROR]** — a brief inline label identifying the specific mistake being corrected.
>
> Unmarked text is carried over from the prior document and is not disputed.
>
> **Overall note:** This document is substantially shorter and less thorough than previous
> attempts, and contains several structural errors serious enough to mislead a reader trying
> to understand the codebase. The corrections below are correspondingly more extensive.

---

### 1. Identification of Notebooks

> **[⚠ CORRECTED — ❌ CRITICAL: Scripts are swapped between notebooks]**
> The prior document correctly identifies *which notebook implements which dynamic* (RL vs.
> replicator), but **completely reverses the Python script each notebook imports from.** The
> RL notebook imports from the `BZforget3` script; the replicator notebook imports from the
> `FLIP7` script. The prior document states the opposite. This is corrected below.

- **Notebook 1: Reinforcement Learning (Roth-Erev Dynamics)**
  - **File:** `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`
  - **Script:** `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`
  - **Identification Key:** Imports `genBS_f7_RothErevExec_full_play_typed`. The function name explicitly references **RothErev**, matching **Appendix B**. The script name contains `BZforget3` (referencing the forgetting mechanism) and `RothErev`.

- **Notebook 2: Replicator Dynamics**
  - **File:** `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`
  - **Script:** `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`
  - **Identification Key:** Imports `genBSreplicatorexec_full_play_typed`. The function name explicitly references **replicatorexec**, matching **Section 2.2**. The script name contains `FLIP7` and `SMARTmutation`, reflecting the replicator variant with smart two-parameter mutation.

> **[➕ ADDED]** The two notebooks import from **two separate Python scripts**, not a shared
> one. Each script contains functions specific to its learning dynamic, though both share a
> large body of analysis and assortment utility functions that are functionally identical.

---

### 2. Detailed Explanation: Replicator Dynamics (Notebook 2)

The Replicator Dynamics model simulates the evolution of strategy profiles at the population
level. Strategy profiles with higher-than-average utility increase in prevalence.

#### Core Model Implementation

- **The Replicator Equation:** `genBSreplicator_single_tstep` implements the discrete
  replicator equation from **Section 2.2**:
  N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) × [U_k(x) - Avg(U_k(i))]
  The line `profile_count_typed[k][x] += profile_count_typed[k][x] * (profile_utility_bytype[k][x] - typesaverage[k])` directly executes this update, followed by non-negativity clamping, renormalization to preserve type totals, and cascade integer rounding.

- **Utility Calculation (U_k(x)):** The function `executilcompute` (the active utility
  function) calculates the expected payoff for strategy profile x for preference type k.
  It iterates through all possible profile pairings, calculates the payoff based on
  `coordination_preferencesf`, weights by assortment multipliers, and subtracts signal cost
  per active dimension.

  > **[⚠ CORRECTED — ❌ `utilcompute` listed alongside `executilcompute` as equivalent]**
  > The prior document states "The function `utilcompute` (and its extension `executilcompute`)
  > calculates the expected payoff." In the active replicator notebook, only `executilcompute`
  > is called. `utilcompute` (without the "exec" prefix) lacks signal cost handling and is used
  > only in older, non-exec replicator variants not called by either active notebook.

- **Mutation:** Implemented in `repmutation_smart`, using `muterandoms10_smart` to determine
  which agents mutate (probability m₀) and which positions in their strategy string change
  (probability m₁ per position), per **Section 2.2**. These are the `_smart` variants; the
  non-smart versions (`muterandoms10`, `repmutation`) are present in the script but not active.

- **Strategy Representation:** `replicatorexec_initialize` sets up `profiles_indexed` (the
  lookup table mapping integer indices to full strategy strings) and structural parameters.
  `execrepconvert` converts the integer-profile representation back to urn format for
  reporting.

  > **[⚠ CORRECTED — ❌ `replicator_initialize` and `reconvert` cited as active]**
  > The prior document names `replicator_initialize` and `reconvert` as the relevant functions.
  > The active replicator uses `replicatorexec_initialize` and `execrepconvert` (the "exec"
  > variants). The non-exec versions lack the attention/signal-cost mechanism and belong to
  > older model variants not called by the active notebook.

  > **[➕ ADDED]** The actual random profile assignment (producing `profile_count_typed`,
  > `profile_count_untyped`, and `agentprofiles`) is performed by `genBSreplicatorexec_k_step`,
  > called after `replicatorexec_initialize`. `genBSreplicator_first_tstep` (an alternative
  > initializer) is **commented out** in the active code.

#### Extensions (Executives & Attention)

- **Executive Attention:** `executilcompute` implements the null-signal masking rule from
  **Section 3.3**: if either agent has a null (0) signal in a dimension, that dimension is
  zeroed for both agents when computing the effective signal they use to look up actions.
  Assortment multipliers, however, use actual (pre-masking) broadcast signals.

- **Signal Cost:** `executilcompute` subtracts `sigcostf` (a negative value) for each
  dimension where an agent broadcasts a non-null signal, implementing the cost c per attended
  dimension from **Section 3.3** and **Section 4.1**.

  > **[⚠ CORRECTED — ❌ `execurnsf` described as dynamically tracking attention]**
  > The prior document states "`execurnsf` tracks whether an agent is attending to a dimension
  > (1) or ignoring it (0)." While this describes what the values represent, `execurnsf` is
  > **not updated during simulation dynamics.** It is initialized to zeros and populated only
  > at **reporting time** by `execrepconvert`, which infers attention from the argmax of signal
  > urns (a non-null modal signal = attending). The attention mechanism itself is implemented
  > through the null-signal masking logic inside `executilcompute`, not through `execurnsf`.

**[➕ ADDED — Convergence check]** The replicator loop checks for early termination every
100 timesteps by comparing `agentprofilesCOPY` to `agentprofiles`. If the distribution is
unchanged, the simulation halts before reaching the maximum `runlength`.

**[➕ ADDED — Inactive replicator functions]** Several functions in the replicator script are
present but not called by the active notebook: `utilcompute`, `replicator_initialize`,
`reconvert` (non-exec repconvert), `genBSreplicator_full_play` (non-exec main loop),
`genBSreplicator_first_tstep` (commented out), `connections_init`, `pairings`, `social_draws`,
`receiver_draws`, `genBS_check_success`. These are legacy implementations superseded by the
exec variants used in the active notebook.

---

### 3. Detailed Explanation: Reinforcement Learning (Notebook 1)

The Reinforcement Learning model (Appendix B) focuses on individual-level urn updates.

#### Core Model Implementation

- **The Urn Mechanism:** `sigurnsf` (signal urns) and `recurnsf` (receiver/action urns)
  represent the per-agent urns described in Appendix B. The urn contents are initialized to
  `inertia = 1` (one ball per option) **in the notebook itself** before the simulation call
  — not inside any initialize function.

  > **[➕ ADDED]** `RothErevExec_initialize` only computes structural parameters (`sigperdimf`,
  > `profilecaps`). It does not touch urn contents.

- **Strategy Selection (Drawing Balls):** `random2picks` implements drawing one ball from
  each urn per agent. The draws are stored in `agentprofilepicks`, a per-agent array holding
  the complete strategy profile employed for the entire timestep. This is the central state
  variable of the RL model for each timestep.

- **Reinforcement (Adding Balls):** `executilcomputeREforget` adds payoff-weighted
  reinforcement to the specific urns that contributed to each interaction:
  `sigurnsf[dim][agent][signal_drawn] += Areinforcement`
  `recurnsf[agent][signal_seen][action_drawn] += Areinforcement`
  Uninvolved urns are not updated, matching Appendix B's selective reinforcement rule.

- **Forgetting Dynamics:** Within `executilcomputeREforget`, forgetting is applied **first**
  (before reinforcement): `sigurnsf[...] *= forgetf`, `recurnsf[...] *= forgetf`. After
  reinforcement, the floor rule raises any urn entry below 1 back to 1.

#### Assortment and Interaction

> **[⚠ CORRECTED — ❌ CRITICAL: Active RL interaction loop is entirely wrong]**
> The prior document describes the RL interaction loop as:
> 1. `social_draws`: Agents broadcast signals.
> 2. `pairings`: Agents are paired (weighted by assortment).
> 3. `receiver_draws`: Agents choose actions based on the partner's signal.
> 4. `executilcomputeREforget`: Payoffs are calculated and urns are updated.
>
> **None of `social_draws`, `pairings`, or `receiver_draws` are called by the active RL
> function `genBS_f7_RothErevExec_full_play`.** These are legacy functions from older
> pair-sampling RL variants. The **actual** active RL loop per timestep is:
> 1. `random2picks`: Each agent draws a strategy profile from their urns → `agentprofilepicks`
> 2. `get_sig_countsRE`: Count current signal distribution (for assortment normalization)
> 3. `find_assort_multipliers`: Compute normalized H(h,x,i) assortment weights
> 4. `executilcomputeREforget`: For all agent pairs, compute payoffs and update urns
>
> This is a critical distinction: the active model computes interactions analytically over
> all agent pairs, not by drawing explicit random pairings per timestep.

- **Assortment (Homophily):** Two functions handle assortment.
  - `execassort_array_create` (**not `assort_array_create`**): Creates a static lookup table
    of raw assortment weights between signal pairs, counting only dimensions where **both**
    agents have a non-null signal. The "exec" variant matters: `assort_array_create` (without
    exec) counts even mutual-null dimensions and is not called by the active RL notebook.
  - `find_assort_multipliers`: Normalizes the raw weights by current population signal
    frequencies to produce H(h,x,i) as defined in Appendix A/B.

**[➕ ADDED — `agentprofilepicks`]** `agentprofilepicks` is the central per-timestep RL
state variable — a 2D array of shape [population_size, profile_length] holding each agent's
drawn strategy for the current timestep. It is produced by `random2picks` and consumed by
`executilcomputeREforget`. It does not persist across timesteps; `random2picks` overwrites it
each round.

**[➕ ADDED — No active mutation in RL]** The `mutateratef` parameter is passed to
`genBS_f7_RothErevExec_full_play` but is **not used** inside that function. Exploration is
provided instead by stochastic urn sampling (`random2picks`) and the floor constraint.

**[➕ ADDED — Convergence check]** The RL loop checks for convergence every **1,000**
timesteps (vs. 100 for the replicator), comparing `agentprofilepicksCOPY` to
`agentprofilepicks`. The longer interval reflects that stochastic draws from stable urns
produce different profiles on successive timesteps even at equilibrium.

**[➕ ADDED — Coordination preferences scaled in RL]** The RL notebook sets
`coordination_preferences` divided by 200 (e.g., 1.5/200 = 0.0075), matching Appendix B
Table 1. This keeps reinforcement increments small relative to initial urn weights of 1,
preventing early interactions from irreversibly dominating urn distributions.

---

### 4. Summary Mapping Table (Corrected)

> **[⚠ CORRECTED]** The prior table listed `pairings` and `connections_init` as the active
> pairing mechanism, and `assort_array_create` (without exec) as the active assortment
> function. Neither is correct for the active notebooks. The table is corrected below.
> `sigdraws` and `recdraws` are also removed from the active model mapping — they belong to
> legacy functions not called by either active notebook.

| Model Component | Paper/Supplement Description | Active Code Subfunction(s) |
|:---|:---|:---|
| **Strategy Profile** | A string of length K+θ | `profiles_indexed`, `repmultipliersf` |
| **Coordination Payoff** | Table 2 / Table 6 / Table 7 | `coordination_preferencesf` (2D: [type, action]) |
| **Replicator Update** | ΔN = N(U − Avg(U)) | `genBSreplicator_single_tstep` |
| **Utility Calculation** | U_k(x) | `executilcompute` (not `utilcompute`) |
| **Mutation** | Probabilities m₀, m₁ | `repmutation_smart`, `muterandoms10_smart` (replicator only; absent from active RL loop) |
| **Reinforcement** | Adding balls to urns | `executilcomputeREforget` (incrementing `sigurnsf`/`recurnsf`) |
| **Forgetting** | Multiply balls by φ | `executilcomputeREforget` (multiplying by `forgetf`, applied before reinforcement) |
| **Urn Drawing** | Drawing one ball per urn | `random2picks` → `agentprofilepicks` |
| **Assortment** | Similarity multiplier H(h,x,i) | `execassort_array_create`, `find_assort_multipliers` |
| **Signal Cost** | Cost c per dimension | `sigcostf` (used in `executilcompute` and `executilcomputeREforget`) |
| **Attention / Null Signal** | Null signal (0) / Ignoring dims | Null-signal masking logic inside `executilcompute` / `executilcomputeREforget`; `execurnsf` at reporting time only |
| **Pairing** | All-pairs analytical interaction | Implicit in `executilcompute` (replicator) and `executilcomputeREforget` (RL) loops over all profile/agent pairs |
| **Random initialization** | Randomly assigned strategy profiles | `genBSreplicatorexec_k_step` (replicator); `inertia=1` in notebook + `random2picks` (RL) |
| **Results / output** | Coordination success metrics | `genBSexec_results` / `genBS_resultsRE`, `simple_stat_001`, `simple_stat_002` |

---

### Block 1: The Strategy Architecture

The paper describes a strategy as a "string" (e.g., ⟨1, BS⟩). The code uses a base-conversion
technique to turn these multidimensional strategies into a single integer index.

- **`repmultipliersf`:** Not explicitly mentioned in the paper, but implements the
  mixed-radix positional weights for the full profile string (signals + actions). Allows
  the code to convert a multi-dimensional coordinate (signal in dim 1, signal in dim 2,
  action for signal combo 0, action for signal combo 1, …) into a unique integer. Essential
  for `profiles_indexed`.

- **`profiles_indexed`:** A lookup table mapping every integer profile index to its full
  strategy vector. `profiles_indexed[i]` returns the complete signal/action string for
  strategy i. Enables O(1) decoding during utility and reporting computations.

- **`sigmultipliersf`:** Similar to `repmultipliersf` but covers only the signal dimensions
  (not action contingencies). Converts a multidimensional signal vector into a single integer
  "combined signal index" used for receiver-urn lookups.

- **`profilecaps`:** An array defining the valid range for each position in the strategy
  string, ensuring mutation and initialization stay within bounds.

---

### Block 2: The Population and Agent State

- **`popt` (`poptf`):** A 1D array of length N where `popt[i]` is the preference type of
  agent i. Fixed throughout each simulation; types never change, only strategies do.

- **`typestotal`:** Count of agents per preference type. Used to maintain constant type sizes
  during replicator renormalization.

- **`profile_count_typed`:** N_{k,t}(x) — a 2D array [type][strategy_index] tracking the
  number of agents of type k using strategy x. The core state variable of the replicator.

- **`profile_count_untyped`:** Σ_k N_{k,t}(x) — summed across types. Acts as M(i) in the
  utility calculation (number of agents playing strategy i regardless of type).

**[➕ ADDED — `agentprofiles`]** A 1D integer array (replicator only) where `agentprofiles[i]`
is the current strategy profile index of agent i. Rebuilt each timestep by
`genBSreplicator_single_tstep` from the updated `profile_count_typed`.

---

### Block 3: The Interaction Mechanics (The "Urns")

> **[⚠ CORRECTED — ❌ `sigdraws` and `recdraws` described as active RL variables]**
> The prior document described `sigdraws` (the "realized" signal) and `recdraws` (the
> "realized" action) as the outputs of the RL drawing process. **These variables are
> produced by `social_draws` and `receiver_draws`, which are NOT called by the active RL
> notebook.** In the active RL model, the equivalent role is played by `agentprofilepicks`,
> produced by `random2picks`. `sigdraws` and `recdraws` exist only in the legacy pair-based
> variants of the simulation.

- **`sigurnsf` (Signal Urns):** A list of 2D arrays, one per dimension:
  `sigurnsf[d][agent][signal_value]`. Stores urn weights for broadcasting each signal value
  in dimension d.

- **`recurnsf` (Receiver Urns):** A 3D array `[agent][signal_received][action_taken]`.
  Stores urn weights for each action given each observed combined signal. A conditional
  strategy — the weight of action a given signal combo s.

- **`agentprofilepicks`:** (**[➕ ADDED]** — not mentioned in prior document.) The key
  per-timestep strategy array in the RL model. Shape [population_size, profile_length].
  Produced by `random2picks` each timestep; holds the specific signal/action combination
  each agent draws from their urns. Passed to `executilcomputeREforget` as the profile each
  agent employs for all interactions that timestep. Overwritten each timestep.

- **`epsilonf`:** A tiny constant (machine epsilon) added to random draws to ensure sampling
  from (0, 1] rather than [0, 1), preventing degenerate zero-index selection.

---

### Block 4: The Dynamics (Replicator vs. Reinforcement)

#### Replicator Logic (Notebook 2)

- **`executilcompute` (not `utilcompute`):** Computes U_k(x). Iterates over all profile
  pairs present in the population, computes payoff weighted by assortment multipliers and
  profile counts, and subtracts signal cost per active dimension. A **population-level**
  calculation.

- **`genBSreplicator_single_tstep`:** Updates the *frequency* of strategies via the
  replicator equation. Also performs cascade integer rounding (converting floating-point
  counts back to integers while preserving type totals) and rebuilds `agentprofiles`.

#### Reinforcement Logic (Notebook 1)

- **`executilcomputeREforget`:** An **individual-level** calculation. For every pair of
  agents, computes payoff from their drawn profiles, then increments their specific signal
  and receiver urn entries directly. Also applies forgetting (before reinforcement) and
  enforces the urn floor of 1 (after reinforcement).

- **`forgetf`:** φ = 0.998 in the RL notebook. Applied multiplicatively to all urn entries
  at the start of each timestep's update. Prevents counts from growing indefinitely and
  allows for strategy shifts over time.

---

### Block 5: The Extensions (Assortment & Attention)

- **`execassort_array_create`:** Builds a static [numsignals × numsignals] matrix of raw
  assortment weights. For each signal pair (i, j), counts the number of non-null dimensions
  they share and applies `base_connection_weights[d]^homophily_factor`. The "exec" prefix
  is significant: **only non-null matching dimensions are counted.**

- **`find_assort_multipliers`:** Normalizes the raw weights by current population signal
  counts, producing the final H(h,x,i) multiplier used to scale payoffs. Called each
  timestep because signal counts change.

- **Attention masking (null-signal rule):** When computing interactions, if either agent
  has a 0 in a signal dimension, that dimension is zeroed for both agents' effective signal.
  This implements Section 3.3: agents not attending treat all others as also broadcasting 0.
  This masking happens inside `executilcompute` and `executilcomputeREforget`, not through
  `execurnsf`.

- **`sigcostf`:** A negative value subtracted from utility/reinforcement for each dimension
  where an agent broadcasts a non-null signal. Applied per active dimension, so attending to
  two dimensions costs 2c. (Section 4.1: "the signal cost c is incurred for each dimension
  attended to.")

- **`genBSpunishf`:** The payoff received when coordination fails. Set to 0 in both active
  notebooks. With this value, failed interactions contribute 0 payoff (before signal cost
  is applied). It is not a "punishment" in the sense of an additional penalty — it is the
  coordination failure payoff that replaces the coordination preference value when agents
  miscoordinate.

---

### Summary of Variable Mapping (Corrected and Extended)

> **[⚠ CORRECTED]** `pairings` and `connections_init` removed as active functions.
> `sigdraws`/`recdraws` removed as active RL variables. `agentprofilepicks` added.

| Variable | Mathematical Concept | Implementation Detail |
|:---|:---|:---|
| `popt` | Preference Type k | Agent-to-type mapping; fixed throughout simulation |
| `sigurnsf` | Sender urns (RL) | Per-dimension urn weights for broadcasting signals |
| `recurnsf` | Receiver urns (RL) | Per-signal-combo urn weights for conditional actions |
| `agentprofilepicks` | Drawn strategy per timestep (RL) | Produced by `random2picks`; consumed by `executilcomputeREforget` |
| `agentprofiles` | Current strategy per agent (Replicator) | Integer profile index; rebuilt each timestep |
| `profile_count_typed` | N_{k,t}(x) | Count of type-k agents using profile x |
| `profile_count_untyped` | M(i) = Σ_k N_{k,t}(x) | Total agents using profile x; used as M(i) in utility |
| `repmultipliersf` | Strategy string encoding | Mixed-radix weights for full profile indexing |
| `assort_multipliers` | H(h,x,i) | Normalized interaction weights; scaled by population signal frequencies |
| `forgetf` | Forgetting factor φ | Multiplicative decay applied to all urns before reinforcement |
| `sigcostf` | Signaling cost c | Per-dimension cost for non-null signal broadcast; negative value |
| `execurnsf` | Attention flags (reporting only) | Inferred at reporting time from urn argmax; not updated during dynamics |
| `genBSpunishf` | Coordination failure payoff | Replaces coordination preference on mismatch; set to 0 in active sweeps |

---

**[➕ ADDED — Functions present in both scripts but not called by either active notebook]**
The following are legacy functions not invoked in the active simulation paths: `connections_init`,
`pairings`, `social_draws`, `receiver_draws`, `genBS_check_success`, `greetings_*`,
`executilcomputeRE` (RL without forgetting), `utilcompute` (replicator without signal cost),
`genBSreplicator_full_play` (non-exec replicator loop), `genBSreplicator_first_tstep`
(commented out). Treating any of these as active leads to incorrect descriptions of the model.

**[➕ ADDED — Key implementation details not in prior document]**
1. **Cascade rounding** in `genBSreplicator_single_tstep`: converts float profile counts
   back to integers via cumulative `np.around` while preserving type totals.
2. **Forgetting occurs before reinforcement** within `executilcomputeREforget`: old weights
   are decayed first; payoffs are added on top.
3. **All-pairs interaction**: both `executilcompute` (replicator) and `executilcomputeREforget`
   (RL) iterate over all agent pairs analytically, not over explicit random pairings.
4. **`flip7zero_correction`**: a numerical safety function in the replicator script that
   randomly reinitializes a type's profile distribution if cascade rounding drives it to
   zero agents. Not mentioned in the paper.
5. **`getalphasignaling`**: a post-processing function that relabels signals across runs so
   that type 0's modal signal is consistently called "alpha," enabling meaningful cross-run
   averaging. Not mentioned in the paper.
