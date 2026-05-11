# Code–Paper Correspondence: Social Identity Signaling Simulations
## Corrected Review of Third-Model Attempt (Qwen 36B 27B)

> **About this document:** This document corrects and supplements the third model's mapping.
> All modifications are marked:
> - **[⚠ CORRECTED]** — a factual error has been fixed.
> - **[➕ ADDED]** — new material not present in the prior document.
> - **[❌ PRIOR ERROR]** — a brief inline label identifying the specific mistake being corrected.
>
> Unmarked text is carried over from the prior document and is not disputed.

---

### 🔍 Notebook Identification

> **[➕ ADDED — Filename note]** The notebook filenames below include the suffix `_short` (e.g.,
> `...-Copy4_short.ipynb`), which does not appear in the actual uploaded files. The correct
> filenames are `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb` and
> `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`. Apart from
> this naming artifact, the identification of dynamics and scripts is correct and notably better
> than earlier attempts — in particular, the two distinct Python scripts are correctly named.

| Notebook File | Learning Dynamic Implemented | Corresponding Documentation |
|:---|:---|:---|
| `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb` | **Discrete Replicator Dynamics** | Main paper, Section 2.2 ("Evolutionary Dynamics") |
| `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb` | **Roth-Erev Reinforcement Learning with Forgetting** | Supplement, Appendix B ("Reinforcement Learning Dynamics") |

---

## 📘 1. Replicator Dynamics Implementation

**Notebook:** `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep...ipynb`
**Core Logic:** Strategies evolve across discrete timesteps based on population-averaged payoffs. Agents inherit strategies from higher-performing members of their preference type. Mutation introduces stochastic variation.

| Code Subfunction | Model Component / Description | Paper/Supplement Reference |
|:---|:---|:---|
| `population_array` | Generates a fixed distribution of preference types across the population. Matches the model's assumption that preferences are exogenously fixed. | Sec 2.2: *"initialized a population of N=1,000 agents with a fixed distribution of preference types"* |
| `replicatorexec_initialize` | Sets up the `profiles_indexed` lookup table (mapping every valid strategy profile to a unique integer), `sigperdim`, and `profilecaps`. Also assigns each agent a random one-hot signal/action urn for structural setup purposes. **[⚠ CORRECTED below — does not by itself produce the random profile assignments used in the main loop]** | Sec 2.2: *"randomly assigned a behavior strategy to each"*; Sec 3.1: strategy profiles as strings of length K+θ |
| `genBSreplicatorexec_k_step` | **The active random profile initializer.** Assigns each agent a uniformly random profile index from all `numprofiles` possibilities, producing the initial `profile_count_typed`, `profile_count_untyped`, and `agentprofiles` arrays used throughout the main loop. | Sec 2.2: *"randomly assigned a behavior strategy to each"* — this is the function that actually performs this step |
| `execassort_array_create` | Computes homophily weights between all possible signal combinations. Counts overlapping **non-null** dimensions only and applies `base_connection_weights**homophily_factor`. | Appendix A: assortment based on signal similarity; S(h,x,j) = (2·d)^h |
| `find_assort_multipliers` | Normalizes assortment weights so they sum to the population size, converting relative similarity into expected interaction frequencies. | Implementation detail: scales H(h,x,i) to match discrete population counts |
| `get_sig_counts` | Tallies how many agents broadcast each multidimensional signal combination, feeding into assortment normalization. | Implementation detail: required for frequency-weighted assortment calculation |
| `executilcompute` | **Core Utility Calculator.** Computes U_k(x) for every strategy profile x of type k. Loops over all coexisting profiles, applies coordination preferences, subtracts `sigcostf` for every non-null signal dimension, and weights interactions by assortment multipliers. | Sec 2.2: utility equation for U_k(x); Sec 3.3: *"signal cost c that an agent incurs if she broadcasts any signal other than 0"* |
| `genBSreplicator_single_tstep` | **Replicator Equation.** Updates strategy counts: `N_new = N_old + N_old * (U_k(x) - Avg(U_k))`. Enforces non-negativity, re-normalizes to preserve type totals, and cascades rounding to assign integer agent counts to profiles. | Sec 2.2: N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) × [U_k(x) - Avg(U_k)]; *"re-normalized so that the number of agents of each preference type remains constant"* |
| `muterandoms10_smart` & `repmutation_smart` | **Mutation Process.** Draws random masks for m₀ (agent selected for mutation) and m₁ (individual strategy elements mutate). Replaces mutated elements with random valid values while preserving unmutated components. | Sec 2.2: *"Each agent is selected for mutation with probability m₀. If selected, each element... is assigned a random value... with probability m₁."* |
| `execrepconvert` | Converts the final profile count arrays back into explicit signal (`sigurnsf`) and receiver (`recurnsf`) arrays for time-series recording and final reporting. | Implementation detail: bridges aggregated replicator state back to per-agent urn format for analysis |
| `genBSreplicatorexec_full_play_typed` | **Main Simulation Loop.** Orchestrates initialization, repeated calls to utility computation, replicator updates, mutation, time-series recording (`time_update`), and snapshot analysis. Handles convergence checking (`agentprofilesCOPY`) every 100 timesteps. | Sec 2.2: *"model dynamics proceeded in discrete timesteps... after adjusting prevalence... re-normalized"* |

> **[⚠ CORRECTED — ❌ `replicatorexec_initialize` conflated with `genBSreplicatorexec_k_step`]**
> The prior document describes `replicatorexec_initialize` as the function that "assigns each
> agent a random deterministic strategy profile." While `replicatorexec_initialize` does set up
> the urn and `profiles_indexed` structure (assigning one-hot random urns per agent), the
> function that actually produces the `profile_count_typed`, `profile_count_untyped`, and
> `agentprofiles` arrays used in the main replicator loop is **`genBSreplicatorexec_k_step`**,
> called immediately after. In the active code, `genBSreplicator_first_tstep` (an alternative
> initializer) is **commented out** and replaced by `genBSreplicatorexec_k_step`. The two
> functions should not be conflated.

**[➕ ADDED — Convergence check]** The active replicator loop checks for early termination
every **100** timesteps by comparing `agentprofilesCOPY` to `agentprofiles`. If the profile
distribution is identical, the simulation halts. This matches the paper's footnote description.

**[➕ ADDED — Inactive functions in the replicator script]** Beyond the functions listed on
line 31 of the prior document (`connections_init`, `pairings`, `social_draws`, `receiver_draws`,
`genBS_check_success`), the following are also present in the replicator script but not called
by the active notebook: `utilcompute` (the non-exec utility function lacking signal cost),
`genBSreplicator_full_play` (the non-exec replicator loop), and `genBSreplicator_first_tstep`
(commented out in the active code). Noting these prevents misattributing them to the active
simulation.

**Unused but Present Functions:** `connections_init`, `pairings`, `social_draws`,
`receiver_draws`, `genBS_check_success` implement explicit pairwise interactions. They are
**not called** in the replicator full-play loop because the model assumes a well-mixed
population where expected payoffs are computed analytically. This matches the paper:
*"utility is calculated based on the assumption that an agent interacts with a large
representative sample of all the other agents in the population."* (Sec 2.2)

---

## 🔵 2. Reinforcement Learning Dynamics Implementation

**Notebook:** `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep...ipynb`
**Core Logic:** Each agent maintains urns for every strategy component. Strategies are drawn stochastically each timestep, agents interact with the population, urns are reinforced by payoffs, and a forgetting factor φ decays counts each round.

| Code Subfunction | Model Component / Description | Paper/Supplement Reference |
|:---|:---|:---|
| `RothErevExec_initialize` | Sets up structural parameters: `sigperdimf` (number of signals per dimension) and `profilecaps` (urn sizes per profile position). **[⚠ CORRECTED below — does NOT initialize urn contents.]** | Appendix B: provides structural scaffolding for the urn system |
| `random2picks` | **Strategy Drawing.** For each agent, samples one value from each signal and receiver urn using cumulative sum thresholds. Populates `agentprofilepicks` for the current timestep. | Appendix B: *"In a single timestep, an agent begins by drawing one ball from each of their urns at random... These draws then determine the strategy profile that the agent employs for the entire timestep"* |
| `get_sig_countsRE` | Counts how many agents drew each signal combination in the current timestep (operates on `agentprofilepicks` rather than static profiles). | Implementation detail: tracks empirical signal distribution for assortment weighting |
| `executilcomputeREforget` | **Core RL Update.** 1. Applies forgetting: `urns *= forgetf` (parameter `BZforget_multiplier` = 0.998). 2. Loops over all agent pairs, computes coordination payoff, subtracts signal costs, weights by assortment `assort_multipliers`. 3. Adds payoff amount to the urns that contributed to the drawn strategy. 4. Floors all urn counts at 1. **Note:** `agentprofilepicks` is passed in as a parameter — this function does not call `random2picks` internally. | Appendix B: forgetting, selective urn reinforcement, assortment weighting, and floor at 1 |
| `execrepconvertRE` | Extracts the most probable signal/action per dimension by taking `argmax` of each urn. Populates `execurnsf` (attention flags) **for reporting only** — not during dynamics. | Implementation detail: RL uses probabilistic urns; deterministic reporting requires extracting modes |
| `genBS_resultsRE` | Computes final coordination statistics (`simple_stat_001`, `simple_stat_002`) adapted for RL urns. | Implementation detail: mirrors `genBSexec_results` but reads from `argmax(urns)` instead of deterministic profiles |
| `genBS_f7_RothErevExec_full_play_typed` | **Main RL Loop.** Orchestrates drawing, assortment weighting, RL updates, time-series recording, and convergence checking (`agentprofilepicksCOPY`) every 1,000 timesteps. | Appendix B: *"We found that having agents select a single strategy profile that they employ for all interactions on a single timestep made this learning dynamic significantly more computationally tractable."* |

> **[⚠ CORRECTED — ❌ `RothErevExec_initialize` does not initialize urn contents]**
> The prior document states "`RothErevExec_initialize`: Initializes `sigurnsf` and `recurnsf`
> with uniform 'inertia' values (1 ball per option)." **This is incorrect.**
> `RothErevExec_initialize` only computes `sigperdimf` and `profilecaps`. It does not touch
> `sigurnsf` or `recurnsf` at all. The urn initialization to `inertia = 1` happens in the
> **notebook itself**, before calling the simulation function:
> `sigurns = [np.ones([population_size, numsignals_perdim[i]]) * inertia for i in ...]`
> `recurns = np.ones([population_size, numsignals, numactions]) * inertia`
> This corresponds to Appendix B's "one ball corresponding to each possible action," but the
> code performing it is in the notebook, not in `RothErevExec_initialize`.

> **[⚠ CORRECTED — ❌ Claim that `genBSpunish` values are swapped during the simulation]**
> The prior document states the main RL loop "Swaps `genBSpunish` values every `snap_length`
> steps as noted in notebook parameters." **No such swap occurs in the active RL simulation.**
> `genBSpunish` is set once in the notebook (to `np.ones((2))*0`) and passed as a fixed
> parameter. The `snap_length` parameter controls snapshot recording intervals, not any change
> to the punishment value. This claim appears to be a misreading of the notebook parameters.

**[➕ ADDED — No active mutation in RL loop]** The `mutateratef` parameter is passed to
`genBS_f7_RothErevExec_full_play` but is **not used** inside that function. There are no
mutation calls in the active RL loop. Exploration comes instead from: (1) stochastic urn
sampling via `random2picks` each timestep, and (2) the floor constraint (minimum urn value = 1)
keeping every option permanently available.

**[➕ ADDED — Convergence check periodicity]** The RL loop checks for convergence every
**1,000** timesteps (vs. 100 for the replicator). The longer interval reflects the fact that
stochastic urn sampling produces different `agentprofilepicks` on successive timesteps even at
a stable urn equilibrium, making 100-step exact-equality checks ineffective.

**[➕ ADDED — Inactive functions in the RL script]** `executilcomputeRE` (the version of the
RL updater without forgetting) is present in the RL script but not called by the active
notebook — `executilcomputeREforget` is used instead.

**[➕ ADDED — Coordination preferences scaled by 1/200 in RL notebook]** The RL notebook
sets `coordination_preferences = (np.array([[1.5, 0, .85], [1.25, 1.5+pa, .85], [0, 0, 1]]))/200`.
The /200 scaling brings payoff values to the range ~0.005–0.0075, matching Appendix B's
Table 1. This keeps reinforcement increments small relative to initial urn weights of 1,
preventing early interactions from irreversibly dominating urn distributions.

---

## 🔗 3. Common & Analytical Functions (Used by Both Notebooks)

| Code Subfunction | Model Component / Description | Paper/Supplement Reference |
|:---|:---|:---|
| `time_update` | Records strategy prevalence over time into 4D arrays (`typed_time`, `signal_time`, etc.). Normalizes by type/signal totals to track evolutionary trajectories. | Implementation detail: enables time-series plotting and convergence analysis |
| `genBSexec_results` / `genBS_resultsRE` | Computes `simple_stat_001` (most likely action pairings) and `simple_stat_002` (high-confidence pairings >75% threshold). Maps directly to the paper's optimality criteria and coordination success metrics. | Sec 4.2: *"We designate a group structure as optimal if... (i) they always successfully coordinate... (ii) that action provides maximal possible sum..."* |
| `getalphasignaling` & `getalphastyped0` | Analyzes signaling alignment post-simulation. Identifies the "alpha" (dominant) signal per dimension for type 0, and **relabels signals across runs so type 0's modal signal is consistently called "alpha,"** enabling meaningful cross-run averaging of signal distributions. Also counts opposition/agreement frequencies and executive attention patterns. | Sec 4.3: *"observing only social signals, without accounting for coordinative behaviors, can lead to overlooking intersectional identities"* |
| `mutual_info` & `average_mutual_info` | Computes Shannon mutual information between transmitted signals and chosen actions across the population. | Sec 4.3: *"Using partial information decomposition... we found that... the synergistic information in the intersectional equilibrium was around 49% of the total information."* |
| `coordination_preferencesf` (Notebook variable) | **2D** matrix of shape [numtypes, numactions] defining payoffs for each type taking each action. **[⚠ CORRECTED below]** | Sec 4.2 Table 7; Appendix B Table 1 |
| `signal_cost`, `homophily_factor`, `mutation_rate`, `BZforget_multiplier` (Notebook variables) | Map directly to model parameters: c (Sec 3.3), h (Appendix A), m₀/m₁ (Sec 2.2), φ (Appendix B). | Explicitly defined in paper/supplement parameter tables and equations |

> **[⚠ CORRECTED — ❌ `coordination_preferencesf` described as a 3D array]**
> The prior document (Table C) describes `coordination_preferencesf` as:
> "3D array: `coordination_preferencesf[type][action][action]` or flattened."
> **This is incorrect.** `coordination_preferencesf` is a **2D array of shape
> [numtypes, numactions]**, where entry [k][a] gives the payoff for preference type k
> coordinating on action a. There is no second action dimension. The payoff p_{xi,k}
> depends on the preference type k and the single action that results from coordination —
> not on two actions simultaneously. The matrix entries correspond directly to the columns
> of the coordination preference tables in the paper (e.g., Table 7's three greeting columns).

---

## ✅ Verification of Completeness

### Paper/Supplement Components → Code Mapping

1. **Generalized BoS Payoffs** → `coordination_preferencesf` in notebooks (2D: [type, action]), applied in `executilcompute` / `executilcomputeREforget`.
2. **Strategy Profile Structure** → Encoded via `repmultipliersf`, `dimactions`, `profilecaps`. Signals + conditional actions stored in `agentprofilepicks` (RL) or `agentprofiles` (Replicator). **[Note: these are distinct variables for distinct models — see correction below.]**
3. **Multidimensional Signaling & Null Attention** → `sigdimensions`, `numsignals_perdim`, `sigmultipliers`. Null signal `0` triggers cost exemption and attention masking in `executilcompute`/`executilcomputeREforget` and reporting functions.
4. **Signal Cost c** → `sigcostf` parameter. Subtracted in utility/RL payoff calculations when `Aprofile[idx] != 0`.
5. **Assortment/Homophily h** → `execassort_array_create`, `find_assort_multipliers`, `get_sig_counts`/`get_sig_countsRE`. Implements H(h,x,i) weighting exactly as in Appendix A/B.
6. **Replicator Dynamics** → `genBSreplicator_single_tstep` implements the exact difference equation and normalization described in Sec 2.2.
7. **Mutation m₀, m₁** → `muterandoms10_smart` & `repmutation_smart` implement the two-stage mutation process verbatim. **Active only in the replicator; absent from the active RL loop.**
8. **RL Urn Structure & Drawing** → Urn content initialized to `inertia=1` **in the notebook**; `RothErevExec_initialize` sets up structural parameters only; `random2picks` performs the per-component draws.
9. **RL Reinforcement & Forgetting φ** → `executilcomputeREforget` applies φ decay, payoff reinforcement, assortment weighting, and floors at 1, matching Appendix B step-by-step.
10. **Optimality & Coordination Metrics** → `genBSexec_results`/`genBS_resultsRE` compute the 75% confidence thresholds and action alignment used to evaluate optimal group structures.

### Code Subfunctions → Paper/Supplement Mapping

- Every function called in the active simulation loops is tied to a mathematical description, simulation procedure, or analytical metric in the provided documents.
- Functions present in the `.py` files but **not called** by the notebooks (`connections_init`, `pairings`, `social_draws`, `receiver_draws`, `genBS_check_success`, `greetings_*`) correspond to older pairwise interaction implementations superseded by the final analytical approach.

**[➕ ADDED — Additional inactive functions not noted above]** The following are also present
in the scripts but not called by either active notebook: `executilcomputeRE` (RL updater without
forgetting), `utilcompute` (replicator utility without signal cost), `genBSreplicator_full_play`
(non-exec replicator loop), `genBSreplicator_first_tstep` (commented out in the active code).

---

## Variable-by-Variable Breakdown

> **Note on Naming Convention**: The trailing `f` in variable names (e.g., `population_sizef`,
> `sigdimensionsf`) is a consistent stylistic choice in the codebase, likely to avoid namespace
> collisions or to standardize Numba function signatures. Conceptually, `varf` ≡ `var`.

---

### 🟦 A. Population & Preference Configuration

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `population_sizef` / `population_size` | N (Sec 2.2, Table 4) | Total number of agents in the simulation. | **Explicit**: Matches N=1000 (replicator) or N=500 (RL) as stated in paper/Appendix B. |
| `numtypesf` / `numtypes` | Preference type count k | Number of distinct coordination preference groups (e.g., 3: Lagashite, Girshite, Akkadian). | **Explicit**: Paper assumes fixed distribution of preference types (Sec 2.2). |
| `poptf` / `popt` | Agent type mapping array | 1D array of length N where `popt[i]` = preference type of agent i. | **Implementation**: Paper describes fixed proportions; code uses this array to partition updates by type. |
| `percent_agents_per_typef` | Type distribution vector | Input to `population_array()`; e.g., `[0.33, 0.33, 0.34]`. | **Explicit**: Matches Table 6/Appendix B population splits. |
| `typestotal` | N_k (count per type) | Tracks total agents per preference type for normalization during replicator updates. | **Implementation**: Required to maintain constant type sizes per Sec 2.2. |

---

### 🟦 B. Signaling Dimensions & Strategy Encoding

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `sigdimensionsf` / `sigdimensions` | K (Sec 4.1) | Number of independent signaling dimensions (e.g., 2 or 3). | **Explicit**: *"agents to signal in up to K dimensions"* (Sec 4.1). |
| `numsignals_perdim` | Signals per dimension | Array like `[2, 2]` meaning 1 null + 1 non-null signal per dimension. Product = `numsignalsf`. | **Explicit**: Paper states *"one non-null signal in each of two dimensions"* → 2 values per dim (0 + 1). |
| `numsignalsf` / `numsignals` | θ (Sec 3.1) | Total unique multidimensional signal combinations = ∏ `numsignals_perdim`. | **Explicit**: θ in Sec 3.1: *"strategy profiles... strings of length 1+θ"*. Here θ = 4 for 2D. |
| `sigmultipliersf` | Signal indexing multipliers | Converts multidimensional signal vectors [s1, s2, ...] into a single integer index via dot product. | **Implementation**: Paper uses mathematical notation; code uses positional multipliers for array storage. |
| `numactionsf` / `numactions` | Action set size | Number of possible coordination actions (e.g., 3 greetings). | **Explicit**: Tables 5–10 show 3 actions. |
| `numprofilesf` / `numprofiles` | Strategy profile space size | Total unique strategy profiles = ∏ `numsignals_perdim` × `numactions`^`numsignals`. | **Implementation**: Encodes the full strategy string length K+θ into a single integer index. |
| `repmultipliersf` | Profile indexing multipliers | Extends `sigmultipliersf` to include receiver dimensions. Maps full strategy profile to a unique integer. | **Implementation**: Necessary for `profiles_indexed` lookups and replicator state tracking. |
| `profiles_indexed` | Strategy profile lookup table | 2D array: `profiles_indexed[profile_idx]` returns the full strategy vector. Used to decode/encode strategies. | **Implementation**: Paper describes strategies as strings; code uses this table for O(1) decoding. |
| `profilecaps` | Valid range per profile component | Array defining the upper bound for each dimension of the strategy string. Used in mutation/initialization. | **Implementation**: Ensures random draws stay within valid signal/action ranges. |
| `agentprofiles` | Current strategy per agent (Replicator only) | Maps each replicator agent index to their active strategy profile integer. Updated by `genBSreplicator_single_tstep` across timesteps. | **Explicit**: Replicator model's per-agent strategy state. **[⚠ CORRECTED — see below]** |
| `agentprofilepicks` | Drawn strategy per agent (RL only) | Populated by `random2picks()` at the start of each timestep. Stores the full signal/action combination drawn from urns. Freshly sampled every timestep. | **Explicit**: *"strategy profile that the agent employs for the entire timestep"* (Appendix B). **[⚠ CORRECTED — see below]** |
| `sigurnsf` | Sender urns | List of 2D arrays, one per dimension: `sigurnsf[d][agent][signal_option]`. Stores weights for broadcasting each signal per dimension. | **Explicit**: Appendix B: *"each agent... having an urn for each location in the string representation of their strategy profile"*. |
| `recurnsf` | Receiver urns | 3D array: `recurnsf[agent][received_signal][action_option]`. Stores weights for conditional actions. | **Explicit**: Matches receiver urns in Appendix B. |
| `execurnsf` | Attention/executive flags (reporting only) | 2D array: `execurnsf[agent][dim]`. Populated at **reporting time** by `execrepconvert`/`execrepconvertRE` from urn argmax. **Not updated during simulation dynamics.** | **[⚠ CORRECTED — see below]** |

> **[⚠ CORRECTED — ❌ `agentprofiles` and `agentprofilepicks` conflated]**
> The prior document lists both under a single entry: "`agentprofiles` / `agentprofilepicks`:
> Current strategy per agent. Maps each agent index to their active strategy profile integer.
> In RL, updated via urn draws each timestep."
> **These are distinct variables for distinct models and should not be merged.** `agentprofiles`
> is a 1D integer array used **only in the replicator** to track each agent's current profile
> index; it is maintained across timesteps and updated by `genBSreplicator_single_tstep`.
> `agentprofilepicks` is a 2D array used **only in the RL model** to store each agent's full
> strategy vector drawn from urns at the start of the current timestep; it is discarded and
> redrawn each timestep. They differ in: what model uses them, their shape, their lifetime
> (persistent vs. per-timestep), and how they are populated.

> **[⚠ CORRECTED — ❌ `execurnsf` described as tracking attention dynamically]**
> The prior document describes `execurnsf` as: "Tracks whether agent attends (1) or ignores (0)
> each dimension." The phrase "Code separates attention tracking for clarity" implies this array
> is maintained during simulation. **It is not.** `execurnsf` is initialized to zeros and
> populated **only at reporting time** by `execrepconvert` (replicator) or `execrepconvertRE`
> (RL), which infer attention from the argmax of signal urns: an agent is flagged as attending
> dimension d if their modal (highest-weight) signal in that dimension is non-null. It has no
> role in the simulation dynamics themselves.

---

### 🟦 C. Payoff Structure & Learning Parameters

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `coordination_preferencesf` | Payoff matrix p_{xi,k} | **2D array of shape [numtypes, numactions]**: entry [k][a] = payoff for type k coordinating on action a. **[⚠ CORRECTED — see prior document's "3D array" error above]** | **Explicit**: Tables 5–10 & Appendix B Table 1. |
| `genBSpunishf` / `genBSpunish` | Failure penalty | Payoff when coordination fails. Usually 0.0 in simulations. Set once in the notebook; **not swapped during the simulation.** | **Explicit**: Generalized BoS assigns 0 for mismatch (Sec 2.1). |
| `sigcostf` / `signal_cost` | c (Sec 3.3) | Cost incurred per attended dimension. Subtracted from payoff when `profile[dim] != 0`. Applied per active signaling dimension. | **Explicit**: *"signal cost c that an agent incurs if she broadcasts any signal other than 0"*. |
| `forgetf` / `BZforget_multiplier` | φ (Appendix B) | Forgetting factor applied multiplicatively to RL urns **at the start** of each timestep's update (before reinforcement). Code uses 0.998. | **Explicit**: *"forgetting is implemented by multiplying... by forgetting factor 0<φ<1"*. |
| `inertia` | Initial urn mass | Base value (1.0) for all urns at t=0. **Set in the notebook, not by `RothErevExec_initialize`.** | **Explicit**: Appendix B: *"Each urn begins a simulation with one ball corresponding to each possible action"*. |
| `epsilonf` / `epsilon` | Numerical stability constant | Added to uniform random draws to map [0,1) → (0,1], preventing zero-probability urn states. | **Implementation**: Not in paper. Standard numerical safeguard for multinomial sampling. |
| `mutateratef` | [m₀, m₁] (Sec 2.2) | Array of mutation probabilities: m₀ = prob agent mutates, m₁ = prob each strategy element mutates. **Passed to but unused by the active RL function.** | **Explicit**: Matches Sec 2.2 mutation description exactly. **Active only in the replicator.** |

---

### 🟦 D. Assortment & Interaction Weighting

| Code Variable | Paper/Supplement Mapping | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `base_connection_weightsf` | Base similarity weights | Array like [1, 2, 4, 8, 16, 32]. Weight for d shared **non-null** signal dimensions = 2^d. | **Explicit**: Appendix A/B: S(h,x,j) = (2·d)^h. Code uses precomputed base values. |
| `homophily_factorf` / `homophily_factor` | h (Appendix A/B) | Exponent applied to connection weights. h=0 → random mixing; h=1 → strong assortment. | **Explicit**: Matches H(h,x,i) formulation. |
| `assort_array` | Precomputed similarity matrix | 2D matrix: `assort_array[sigA][sigB]` = weight based on overlapping non-null dimensions. Built once before the main loop. | **Implementation**: Paper defines H mathematically; code precomputes for O(1) lookup. |
| `assort_sig_counts` | N_j per signal | Frequency count of each signal combination in the population. Updated each timestep. | **Implementation**: Required to normalize assortment weights to population size. |
| `assort_multipliers` | H(h,x,i) scaled | Normalized interaction weights. `assort_multipliers[i][j] = assort_array[i][j] × (N / Σ_k assort_array[i][k] × count_k)`. Used to scale payoffs in utility/RL updates. | **Explicit**: Direct implementation of Appendix A/B normalization formula. |

---

### 🟦 E. Replicator Dynamics State Variables

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `profile_count_typed` | N_{k,t}(x) | 2D array: agents per strategy profile, partitioned by preference type. | **Explicit**: Matches Sec 2.2 equation variables exactly. |
| `profile_count_untyped` | Σ_k N_{k,t}(x) | 1D array: total agents per profile across all types. Used for assortment & utility calculation as M(i). | **Implementation**: Aggregation step required for population-level payoff averaging. |
| `profile_utility_bytype` | U_k(x) | 2D array: expected payoff for each profile-type combination. | **Explicit**: Directly implements Sec 2.2 utility equation. |
| `typesaverage` | Avg(U_k) | 1D array: mean utility across all present profiles for each preference type. | **Explicit**: Denominator in replicator difference equation. |
| `mutations10`, `newprofiles10`, `muteprofilestrings10` | Batched mutation masks | Pre-generated random arrays for 10 timesteps ahead. `mutations10` = m₀ masks, `newprofiles10` = random profile targets, `muteprofilestrings10` = m₁ element masks. | **Implementation**: Not in paper. Vectorization trick to avoid repeated RNG calls inside the tight loop. |

---

### 🟦 F. Reinforcement Learning Urn & Update Variables

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `agentprofilepicks` | Drawn strategy per agent | Populated by `random2picks()`. Stores the exact signal/action combination drawn from urns for the current timestep. Passed as a parameter to `executilcomputeREforget` — not generated inside it. | **Explicit**: *"drawing one ball from each of their urns... determine the strategy profile"* (Appendix B). |
| `picks10` | Pre-generated draw masks | 3D array of uniform random numbers used to sample urns via cumulative thresholding. Refreshed every 10 timesteps. | **Implementation**: Batched RNG for performance. Not in paper. |
| `Areinforcement`, `Breinforcement` | Payoff adjustment amount | Computed as `assort_weight * (coordination_payoff + sig_cost)`. Added only to urns that contributed to the interaction. | **Explicit**: *"if an agent receives a non-zero payoff the urns that contributed to their actions are reinforced"*. |
| `forgetf` (applied in `executilcomputeREforget`) | φ decay | `sigurnsf *= forgetf`, `recurnsf *= forgetf`. Applied **before** reinforcement. Followed by `max(urn, 1)` floor. | **Explicit**: Matches Appendix B forgetting & floor rule exactly. |

---

### 🟦 G. Simulation Control, Time-Series & Analysis Arrays

| Code Variable | Paper/Supplement Equivalent | Functional Role | Explicit vs Implementation Detail |
|:---|:---|:---|:---|
| `runlengthf` | T (Table 4) | Total simulation timesteps (e.g., 4×10^4). | **Explicit**: Matches paper parameter table. |
| `record_intervalf`, `signal_snapshotsf` | Logging frequency | How often to record time-series data and strategy snapshots. | **Implementation**: Not in paper. Standard for generating trajectory data. |
| `typed_time`, `signal_time` | Strategy prevalence over time | 4D arrays storing raw urn/profile counts per timestep. Used for convergence tracking. | **Implementation**: Paper shows equilibrium plots; code records full trajectories. |
| `typed_time_norm`, `signal_time_norm` | Normalized probabilities | Same as above but divided by type/signal totals. Ensures comparability across runs. | **Implementation**: Numerical normalization for plotting. |
| `simple_stat_001`, `simple_stat_002` | Coordination success matrices | `001` = most likely action pairings; `002` = pairings with >75% confidence. Used to validate optimality criteria. | **Explicit**: Implements Sec 4.2 optimality check. |
| `socialsig_aggregate`, `action_aggregate` | Empirical frequency counts | Tallies signals broadcast and actions taken by type. Input to post-hoc analysis. | **Implementation**: Required for `getalphasignaling()` and mutual information calculations. |
| `multidimsignalscount0`, `oppositiondisruns`, `agreeruns`, `execsruns` | Alignment & attention metrics | Tracks how often opposition/agree types match/mismatch type 0's alpha signal, and executive attention patterns. | **Explicit**: Sec 4.3/Appendix analysis on signaling and intersectional identity. |
| `mutuinfo` / `averagemutuinfo` | Shannon mutual information | Computes I(S;A) between signals and actions. Used to quantify synergistic/intersectional information. | **Explicit**: Sec 4.3: *"synergistic information in the intersectional equilibrium was around 49%"*. Code implements base MI. |

---

### 🔍 Critical Implementation Details Not Explicitly Stated in the Paper

1. **Cascade Rounding (`genBSreplicator_single_tstep`)**:
   After applying the replicator equation, profile counts become floats. The code uses cumulative floating-point accumulation combined with `np.around` to assign integer agent counts while preserving type totals. This prevents population drift.

2. **Attention Masking in Utility/RL Updates**:
   When computing payoffs, if either agent has `0` in a signal dimension, that dimension is zeroed out for both agents in the interaction. This implements Sec 3.3: *"When an agent does not attend to signals... she interacts with all other agents as if they had also broadcast 0."* Assortment multipliers use the actual (pre-masking) broadcast signals, since homophily depends on what is physically broadcast.

3. **Batched Mutation & RNG (`mutations10`, `picks10`)**:
   The code pre-generates 10 timesteps of random masks to avoid Python/Numba overhead inside the main loop. Functionally identical to per-timestep generation, but optimized for HPC execution.

4. **Assortment Normalization (`find_assort_multipliers`)**:
   The paper's H(h,x,i) formula yields relative weights. The code scales them so `sum(assort_multipliers[i]) = N`, converting similarity into expected interaction counts.

5. **Urn Floor Enforcement**:
   After forgetting decay (and after reinforcement), any urn count < 1 is set to 1. This preserves exploration probability, exactly matching Appendix B.

6. **Time-Series Dual Tracking**:
   Code records both raw counts (`*_time`) and normalized probabilities (`*_time_norm`). The paper only shows normalized equilibrium states, but dual tracking enables sensitivity analysis and convergence diagnostics.

7. **[➕ ADDED] Forgetting Order**:
   Within `executilcomputeREforget`, forgetting (multiply by φ) occurs **before** reinforcement within the same function call. Old urn weights are decayed first; new payoffs are added on top of the decayed values. This ordering is not stated in Appendix B but is consequential for dynamics.

8. **[➕ ADDED] All-Pairs Interaction in RL**:
   `executilcomputeREforget` iterates over all agent pairs (not random pairs), so each agent effectively interacts with every other agent in the population on each timestep. This is consistent with Appendix B's note on tractability and with the replicator's "large representative sample" assumption.

---

### ✅ Summary of Fidelity

- **Every mathematical variable** in the paper (N, K, θ, c, φ, h, U_k(x), m₀, m₁, H(h,x,i)) has a direct code counterpart.
- **Every simulation step** (initialization, assortment weighting, payoff computation, replicator update/RL reinforcement, forgetting, mutation, time-series logging) is explicitly mapped to the described dynamics.
- **Implementation gaps** are strictly computational: array indexing multipliers, batched RNG, cascade rounding, dual time-series normalization, and attention masking logic. None alter the theoretical model.
- **Legacy functions** (`connections_init`, `pairings`, `social_draws`, `receiver_draws`, `genBS_check_success`, `greetings_*`, `executilcomputeRE`, `utilcompute`, `genBSreplicator_full_play`, `genBSreplicator_first_tstep`) are present in the scripts but not called by either active notebook's primary simulation path.
