Based on the provided files, I have identified which notebook corresponds to which learning dynamic and will provide a detailed explanation of how the code subfunctions implement the model described in the paper and supplement.

### 1. Identification of Notebooks and Learning Dynamics

There are two Jupyter notebooks provided in the text. Based on their filenames and the functions they import from the shared Python script (`FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`), they implement the following dynamics:

| Notebook Filename                                                                    | Learning Dynamic                       | Primary Simulation Function Called      |
|:------------------------------------------------------------------------------------ |:-------------------------------------- |:--------------------------------------- |
| **`genBS_v0055k05_BZforget3_assort_f7RothErevExec_sweep_Merced_topA-Copy34.ipynb`**  | **Reinforcement Learning (Roth-Erev)** | `genBS_f7_RothErevExec_full_play_typed` |
| **`genBS_v0055k05_BZforget3_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`** | **Discrete Replicator Dynamics**       | `genBSreplicatorexec_full_play_typed`   |

The shared Python script contains functions for both dynamics. I will map the code subfunctions to the model descriptions provided in the **Paper** (for Replicator Dynamics) and **Appendix B of the Supplement** (for Reinforcement Learning).

---

### 2. Reinforcement Learning (Roth-Erev) Implementation

**Notebook:** `genBS_v0055k05_BZforget3_assort_f7RothErevExec_sweep_Merced_topA-Copy34.ipynb`
**Model Reference:** Supplement Appendix B ("Reinforcement Learning Dynamics")

This notebook implements the **Roth-Erev reinforcement learning** model where agents maintain "urns" (distributions over actions) and update them based on past payoffs.

#### Key Code Subfunctions & Model Connections:

**A. `genBS_f7_RothErevExec_full_play_typed` (Wrapper & Main Loop)**

* **Code:** Calls `genBS_f7_RothErevExec_full_play`. Initializes `sigurnsf` (Signal Urns) and `recurnsf` (Response Urns) with `inertia`.
* **Model Connection:** Implements the "urns" concept described in the supplement. Each agent has an urn for each dimension of their strategy profile. Initially, urns are initialized with `inertia` balls (representing equal probability over all actions).
* **Subfunction:** `random2picks`:
  * **Code:** `random2picks(randoms4pick, sigdimensionsf, sigurnsf, recurnsf, population_sizef, agentprofilepicks, profile_length, profilecaps, numsignalsf)`
  * **Model Connection:** Implements the **Action Selection** step. For each agent, it draws balls from their signal urns (`sigurnsf`) and response urns (`recurnsf`) to determine their broadcast signal and chosen action for the timestep. This corresponds to the supplement's description: "In the first urn a null ball and a blue ball...".
* **Subfunction:** `executilcomputeREforget`:
  * **Code:** Updates `sigurnsf` and `recurnsf` by adding reinforcement (`+= Areinforcement`) and applying forgetting (`*= forgetf`).
  * **Model Connection:** Implements the **Learning/Update Rule**.
    * **Reinforcement:** `sigurnsf[idx...][popdex][dimdex] += Areinforcement`. If an agent receives a positive payoff from an action, the corresponding urn (ball) is reinforced (more balls added). This matches the supplement: "reinforcements can be thought of as adding additional balls of the type that was drawn".
    * **Forgetting:** `sigurnsf[idx...][popdex][dimdex] *= forgetf`. At the end of the step, ball counts are multiplied by a factor ($\phi < 1$). This matches the supplement: "forgetting is implemented by multiplying the number of balls... by a forgetting factor".
    * **Flooring:** `if sigurnsf[...] < 1: sigurnsf[...] = 1`. Ensures urns don't empty out completely.

**B. `greetings_check_success` & `genBS_check_success`**

* **Code:** Iterates through pairs (`adex`, `bdex`), checks if actions match preferences (`adraw == btype`), and calculates reinforcement.
* **Model Connection:** Implements the **Payoff/Coordination Logic**.
  * `if adraw == btype`: Reinforces success (positive payoff).
  * `else`: Punishes failure (negative payoff).
  * **Assortment:** `assort_array_create` and `find_assort_multipliers` calculate `Ahfmultiplier` and `Bhfmultiplier`. This implements the **Assortment** mechanism from the paper/supplement, where connection weights depend on signal similarity (e.g., $base\_connection\_weights[floor(dimcount)]**homophily\_factorf$).

**C. `execrepconvertRE`**

* **Code:** Converts the internal urn representations (`sigurnsf`, `recurnsf`) back into a profile representation for reporting.
* **Model Connection:** Allows the simulation to track the **Strategy Profile** (the specific string of signals/actions) even though learning happens at the urn level.

---

### 3. Discrete Replicator Dynamics Implementation

**Notebook:** `genBS_v0055k05_BZforget3_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`
**Model Reference:** Paper Section 2.2 ("Evolutionary Dynamics")

This notebook implements the **Discrete Replicator Dynamics** where strategy profiles (integers) are updated based on their relative utility compared to the population average.

#### Key Code Subfunctions & Model Connections:

**A. `genBSreplicatorexec_full_play_typed` (Wrapper & Main Loop)**

* **Code:** Calls `genBSreplicatorexec_full_play`. Initializes `execurnsf` and `profiles_indexed`.
* **Model Connection:** Initializes the population with random strategy profiles. `profiles_indexed` maps integers to the string of signals/actions (e.g., `sigdimensionsf + numsignalsf` length). This matches the paper: "initially... randomly assigned a behavior strategy".
* **Subfunction:** `genBSreplicator_single_tstep`:
  * **Code:** `profile_count_typed[idx...][idx...] += profile_count_typed[idx...][idx...] * (profile_utility_bytype[idx...][idx...] - typesaverage[idx...])`
  * **Model Connection:** Implements the **Replicator Update Rule**: $n_{x,t+1} = n_{x,t} + n_{x,t}(U_x - \bar{U})$. The count of agents using strategy $x$ increases if $U_x$ is above the average utility for their type.
  * **Normalization:** `profile_count_typed[idx...] = profile_count_typed[idx...]*(typestotal[idx...]/np.sum(...))`. Ensures population size remains constant per type.
* **Subfunction:** `utilcompute` / `executilcompute`:
  * **Code:** Calculates `profile_utility_bytype` by iterating over all strategy pairs, calculating payoff based on `coordination_preferencesf` and `base_connection_weightsf`.
  * **Model Connection:** Implements the **Utility Calculation** from the paper (Eq 3 & 4). It sums the payoffs weighted by the prevalence of opponent strategies ($p_{xi,k}$ and $M(i)$). It calculates assortment multipliers (`hfmultiplier`) based on how many signal dimensions match.

**B. `muterandoms10` & `repmutation`**

* **Code:** Applies mutation to the `profile_count_typed` and `profile_count_untyped`.
* **Model Connection:** Implements **Mutation**.
  * `muterandoms10`: Generates random mutations based on `mutateratef`.
  * `repmutation`: Updates counts. If a profile is mutated (`mutations[idx...] == 1`), it is replaced by a new profile from `newprofiles`. This matches the paper's description: "agents' strategy profiles is not copied faithfully, or 'mutates'".

**C. `repconvert`**

* **Code:** Converts the integer `agentprofiles` back into the `sigurnsf` and `recurnsf` array format for data logging.
* **Model Connection:** Translates the discrete strategy profile (integer index) back to the detailed signal/action urn structure for analysis (e.g., calculating mutual information).

**D. `time_update`**

* **Code:** Aggregates `recurnsf` into `typed_time` and `signal_time`.
* **Model Connection:** Implements the **Time Series Recording**. It records the average frequency of actions given specific signals over time, matching the data collection requirements for the results shown in the notebooks.

---

### 4. Summary of Model Components in Code

To ensure no component of the paper's description is disconnected and all code subfunctions are mapped:

| Model Component (Paper/Supplement)                         | Code Subfunction / Variable                                                 | Description of Connection                                                                                              |
|:---------------------------------------------------------- |:--------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------- |
| **Preferential Types** (Ummians, Kishus, Akkadians)        | `popt` / `type0` / `coordination_preferencesf`                              | Defines the population composition and payoff matrices (e.g., Table 5 in paper).                                       |
| **Multidimensional Signals** (2 dimensions, 2 signals/dim) | `sigdimensionsf`, `sigmultipliers`, `sigurnsf`                              | Allows broadcasting signals across dimensions. `sigurnsf` holds probabilities for each signal dimension.               |
| **Conditional Actions** (Action based on signal)           | `recurnsf` (RL) / `profiles_indexed` (Replicator)                           | Stores the strategy: "If Signal 1, do Action 0". RL stores this as urns; Replicator stores as integer profile indices. |
| **Assortment** (Connection based on signal similarity)     | `assort_array_create`, `find_assort_multipliers`, `base_connection_weights` | Calculates the weight of interaction based on how many signal dimensions match (e.g., homophily).                      |
| **Signal Cost**                                            | `signal_cost` / `sigcostf`                                                  | Deducted from payoff if a non-null signal is used in RL (`executilcomputeREforget`).                                   |
| **Forgetting**                                             | `forgetf` (RL)                                                              | Multiplies urn counts by a decay factor (Appendix B).                                                                  |
| **Mutation**                                               | `muterandoms10`, `repmutation` (Replicator)                                 | Randomly changes strategy profile components with probability `mutaterate` (Paper).                                    |
| **Replicator Dynamics**                                    | `genBSreplicator_single_tstep`, `utilcompute`                               | Updates strategy counts based on utility advantage over average (Paper Eq 1).                                          |
| **Reinforcement Learning**                                 | `executilcomputeREforget`, `random2picks`                                   | Updates urn probabilities based on payoff history (Supplement Appendix B).                                             |
| **Results Metrics**                                        | `time_update`, `genBS_results`                                              | Calculates frequencies of greetings, mutual information, and success rates.                                            |

This analysis confirms that both notebooks utilize the shared Python script to implement either the Reinforcement Learning dynamics (Notebook 1, using Roth-Erev urns) or the Replicator Dynamics (Notebook 2, using strategy counts), mapping directly to the descriptions provided in the paper and supplement.

This detailed breakdown correlates every variable and function in the provided Python script (`FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`) and the two Jupyter notebooks directly to the **Paper** (Sections 2.1–4.4) and **Supplement** (Appendix A & B).

The code implements two distinct simulation engines within the same script structure:

1. **Replicator Dynamics Engine:** Used by `genBSreplicator_*` functions (Matches **Paper Section 2.2**).
2. **Reinforcement Learning (Roth-Erev) Engine:** Used by `genBS_f7_RothErevExec_*` functions (Matches **Supplement Appendix B**).

---

### 1. Global Simulation Parameters & State Space

These variables are defined in the **Jupyter Notebooks** (Cells 2) and passed into the `.py` functions. They define the "world" of the simulation.

| Variable                   | Notebook Definition                 | Code Usage                           | Paper/Supplement Correlation                | Implementation Detail                                                                                           |
|:-------------------------- |:----------------------------------- |:------------------------------------ |:------------------------------------------- |:--------------------------------------------------------------------------------------------------------------- |
| `population_size`          | `1000` or `500`                     | Passed to `population_array`         | **Paper:** $N$ (Eq 2)                       | Defines the total number of agents.                                                                             |
| `numtypes`                 | `3` (Ummians, Kishus, Akkadians)    | Passed to `type0`, `type0opposition` | **Paper:** Section 4 (Mesopotamian Framing) | Defines the number of preference types.                                                                         |
| `sigdimensions`            | `2`                                 | Passed to `sigdimensionsf`           | **Paper:** Section 4.1 (Multidimensional)   | Number of independent signaling dimensions (e.g., color + shape).                                               |
| `numsignals_perdim`        | `[2, 2]`                            | Passed to `sigmultipliers`           | **Paper:** Section 3.1 (Signals)            | Number of signals available per dimension.                                                                      |
| `numsignals`               | `4` (Product of dims)               | Passed to `numsignalsf`              | **Paper:** Section 3.1                      | Total unique signal combinations ($2 \times 2 = 4$).                                                            |
| `sigmultipliers`           | `[1, 2, 4, 8]` (Cumulative Product) | Passed to `repconvert`               | **Paper:** Section 3.1 (Profile String)     | Maps a multi-dimensional signal (e.g., Red + Square) to a single integer index for array storage.               |
| `repmultipliers`           | `[1, 2, 4, 8, ...]`                 | Passed to `repmutation`              | **Paper:** Section 2.2 (Strategy Profile)   | Maps a full strategy profile (Signal + Action) to a single integer index for Replicator Dynamics.               |
| `base_connection_weights`  | `[1, 2, 4, 8, ...]`                 | Passed to `connections_init`         | **Paper:** Section 3.3 (Assortment)         | Weighting factor for assortment. $H(h, x, i)$ formula in Supplement.                                            |
| `homophily_factor`         | `0` to `1`                          | Passed to `assort_array_create`      | **Paper:** Section 3.3 (Assortment)         | Exponent applied to connection weights ($base\_connection\_weights^{homophily\_factor}$).                       |
| `coordination_preferences` | `[[1, 1, 0.5], ...]`                | Passed to `utilcompute`              | **Paper:** Table 5, 7 (Payoff Matrix)       | Defines the payoff matrix ($p_{xi,k}$) for each preference type and action.                                     |
| `genBSpunish`              | `0` or negative                     | Passed to `greetings_check_success`  | **Paper:** Section 2.2 (Coordination)       | Penalty value when coordination fails.                                                                          |
| `signal_cost`              | `-0.0005`                           | Passed to `executilcompute`          | **Paper:** Section 3.3 (Cost)               | Subtracted from payoff if a non-null signal is broadcast.                                                       |
| `forgetf`                  | `0.998`                             | Passed to `executilcomputeREforget`  | **Supplement:** Appendix B (RL)             | Decay factor applied to urn values at end of timestep.                                                          |
| `mutation_rate`            | `[0.01, 0.1]`                       | Passed to `repmutation`              | **Paper:** Section 2.2 (Mutation)           | Probability of mutation ($m_0$) and probability of bit-flip ($m_1$).                                            |
| `runlength`                | `4*10^4`                            | Passed to `genBS_full_play`          | **Paper:** Table 4 (Simulation Length)      | Total number of timesteps to run.                                                                               |
| `record_interval`          | `runlength`                         | Passed to `time_update`              | **Paper:** Table 4 (Sampling)               | Frequency at which time-series data is recorded.                                                                |
| `runid`                    | `0`                                 | Passed to `genBS_results`            | **Paper:** Table 4 (Replication)            | Identifier for tracking specific simulation runs.                                                               |
| `epsilon`                  | `float_info.epsilon`                | Passed to `social_draws`             | **Code:** Numerical Stability               | Small value added to random numbers to prevent division by zero or selection bias in probability distributions. |

---

### 2. Initialization Functions

These functions set up the initial state of the population.

| Function                      | Code Subfunctions                                              | Variables                                            | Paper/Supplement Correlation               | Implementation Detail                                                                                                         |
|:----------------------------- |:-------------------------------------------------------------- |:---------------------------------------------------- |:------------------------------------------ |:----------------------------------------------------------------------------------------------------------------------------- |
| `population_array`            | `population_array(population_sizef, percent_agents_per_typef)` | `popt`, `cumsum_papt`                                | **Paper:** Section 2.2 (Initialization)    | Assigns agent types based on percentages (e.g., `0.5 + beta`).                                                                |
| `replicator_initialize`       | `replicator_initialize(rngf, repmultipliersf, ...)`            | `sigurnsf`, `recurnsf`, `profiles_indexed`           | **Paper:** Section 2.2 (Init)              | Initializes **Replicator** state. Creates `profiles_indexed` to map integers to signal/action strings.                        |
| `replicatorexec_initialize`   | `replicatorexec_initialize(epsilonf, rngf, ...)`               | `execurnsf`, `profiles_indexed`                      | **Paper:** Section 4.1 (Executives)        | Initializes **Replicator** state with Executive attention (`execurnsf`). Tracks which dimensions agents *attend* to (0 or 1). |
| `RothErevExec_initialize`     | `RothErevExec_initialize(epsilonf, rngf, ...)`                 | `sigperdimf`, `profilecaps`                          | **Supplement:** Appendix B (Init)          | Initializes **RL** state. Sets up `profilecaps` (array lengths) for urns.                                                     |
| `random2picks`                | `random2picks(randoms4pick, ...)`                              | `agentprofilepicks`                                  | **Supplement:** Appendix B (Action Select) | Selects actions based on urn probabilities (`sigurnsf`, `recurnsf`). Maps continuous probabilities to discrete actions.       |
| `genBSreplicator_first_tstep` | `genBSreplicator_first_tstep(...)`                             | `typestotal`, `profile_count_typed`, `agentprofiles` | **Paper:** Section 2.2 (Init)              | Initializes Replicator counts. Maps initial random profiles to integer counts.                                                |

---

### 3. The Core Learning Loop (Step-by-Step)

This is the heart of the simulation. The code diverges here into **Replicator** or **RL** logic.

| Variable / Function            | Replicator Logic                            | RL Logic (Roth-Erev)                   | Paper/Supplement Correlation  | Implementation Detail                                                                                                    |
|:------------------------------ |:------------------------------------------- |:-------------------------------------- |:----------------------------- |:------------------------------------------------------------------------------------------------------------------------ |
| `agentprofilepicks`            | `genBSreplicator_first_tstep` returns this  | `random2picks` generates this          | **Code:** Strategy Profile    | Stores the integer index of the strategy chosen for the current timestep.                                                |
| `profile_count_typed`          | `genBSreplicator_single_tstep` updates this | N/A (RL updates urns directly)         | **Paper:** Eq 1               | Number of agents of type $k$ using strategy $x$. Updated via replicator equation.                                        |
| `profile_count_untyped`        | `genBSreplicator_single_tstep` updates this | N/A                                    | **Paper:** Eq 1               | Total agents using strategy $x$ (sum across types).                                                                      |
| `final_sigurns`                | N/A                                         | `executilcomputeREforget` returns this | **Supplement:** Appendix B    | Stores the probability distributions (urns) for each agent's signals/actions.                                            |
| `final_recurns`                | N/A                                         | `executilcomputeREforget` returns this | **Supplement:** Appendix B    | Stores the probability distributions (urns) for each agent's actions.                                                    |
| `execurnsf`                    | N/A                                         | `executilcomputeREforget` returns this | **Code:** Executive Attention | Tracks which dimensions agents attended to (1) or ignored (0). Used for utility calculation.                             |
| `genBSreplicator_single_tstep` | `profile_count_typed` update loop           | N/A                                    | **Paper:** Eq 1               | **Math:** $N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) \times [U_k(x) - Avg(U_k)]$. Updates population counts.                |
| `executilcompute`              | `profile_utility_bytype` calc               | N/A                                    | **Paper:** Eq 3 & 4           | Calculates expected utility for Replicator agents.                                                                       |
| `executilcomputeREforget`      | N/A                                         | `sigurnsf`, `recurnsf` update loops    | **Supplement:** Appendix B    | **Math:** $P(action                                                                                                      |
| `repmutation`                  | `repmutation` function                      | N/A                                    | **Paper:** Section 2.2        | Applies mutation to `profile_count_typed` and `profile_count_untyped`. `newprofiles` replaces old ones.                  |
| `muterandoms10`                | `muterandoms10` function                    | N/A                                    | **Paper:** Section 2.2        | Generates random mutation masks. `mutateratef` determines probability of changing a bit.                                 |
| `muterandoms10_smart`          | `muterandoms10_smart` function              | N/A                                    | **Code:** Smart Mutation      | Implements two rates: $m_0$ (agent mutation) and $m_1$ (bit flip). Used by `repmutation_smart`.                          |
| `repmutation_smart`            | `repmutation_smart` function                | N/A                                    | **Code:** Smart Mutation      | If a mutation occurs (`mutations[idx] == 1`), calculates `new_profile` using `repmultipliersf` and `muteprofilestrings`. |
| `time_update`                  | `typed_time`, `signal_time`                 | `typed_time`, `signal_time`            | **Paper:** Table 4 (Data)     | Aggregates `recurnsf` into time-series matrices for plotting. Normalizes counts to probabilities.                        |

---

### 4. Utility & Payoff Calculation

This section determines how successful an interaction is.

| Variable / Function         | Description                         | Paper/Supplement Correlation           | Implementation Detail                                                                                                                |
|:--------------------------- |:----------------------------------- |:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------ |
| `assort_array_create`       | Creates `assort_array`              | **Paper:** Section 3.3                 | Calculates $H(h, x, i)$. Checks if signals match in `sigdimensions`. Uses `base_connection_weights` and `homophily_factor`.          |
| `execassort_array_create`   | Creates `execassort_array`          | **Paper:** Section 4.1                 | Modified assortment logic. Only counts matches where `signals_indexed` is not 0 (null). Accounts for executive attention.            |
| `find_assort_multipliers`   | Calculates `assort_multipliers`     | **Paper:** Section 3.3                 | Converts raw assortment weights into multipliers based on population counts (`assort_sig_counts`). Normalizes by `population_sizef`. |
| `greetings_check_success`   | Checks if `adraw == btype`          | **Paper:** Section 2.2 (Greeting Game) | Simple coordination. If Action A matches Type B, reinforce. Else punish.                                                             |
| `genBS_check_success`       | Checks if `adraw == bdraw`          | **Paper:** Section 2.2 (BoS Game)      | Bach/Stravinsky coordination. If Action A matches Action B, reinforce. Else punish.                                                  |
| `utilcompute`               | Calculates `profile_utility_bytype` | **Paper:** Eq 3                        | Calculates utility for Replicator. Iterates over `profiles_indexed`. Uses `assort_multipliers`.                                      |
| `executilcompute`           | Calculates `profile_utility_bytype` | **Paper:** Eq 3 (Exec Version)         | Same as `utilcompute` but includes `sigcostf`.                                                                                       |
| `executilcomputeRE`         | Calculates reinforcement            | **Supplement:** Appendix B             | Calculates payoff for RL agents. Adds `coordination_preferencesf` and `sigcostf` to reinforcement values.                            |
| `sigcostf`                  | Subtracts signal cost               | **Paper:** Section 3.3                 | If agent broadcasts non-null signal, payoff is reduced by `c`.                                                                       |
| `poptf`                     | `populations`                       | **Paper:** Section 2.2                 | Agent types (`Ummians`, `Kishus`, etc.). Used to determine `coordination_preferencesf` index.                                        |
| `coordination_preferencesf` | Payoff matrix                       | **Paper:** Table 5, 7                  | Defines payoff for specific Type-Action pairs (e.g., Umma + Umma Greeting = $1+\alpha$).                                             |

---

### 5. Assortment & Pairing

This section determines who interacts with whom.

| Variable / Function | Description           | Paper/Supplement Correlation | Implementation Detail                                                                                                       |
|:------------------- |:--------------------- |:---------------------------- |:--------------------------------------------------------------------------------------------------------------------------- |
| `connections_init`  | Creates `conweightsf` | **Paper:** Section 3.3       | Calculates connection weights based on signal similarity (`popidf`). Uses `base_connection_weights` and `homophily_factor`. |
| `pairings`          | Creates `pairsf`      | **Paper:** Section 3.3       | Implements weighted random pairing. Uses `randperm01f` (permutation of agents) and `agpick` (cumulative sum selection).     |
| `receiver_draws`    | Calculates `recdraws` | **Paper:** Section 3.1       | Determines actions based on `recurnsf` and partner's signal. Uses `sigmultipliersf` to aggregate multi-dimensional signals. |
| `social_draws`      | Calculates `sigdraws` | **Paper:** Section 3.1       | Determines signals based on `sigurnsf`. Uses `cumsum` and `csrand` (random value) to sample from urn probabilities.         |
| `popidf`            | `popt` array          | **Paper:** Section 3.3       | The "identity" of the population used for assortment calculations.                                                          |

---

### 6. Results & Analysis Functions

These functions process the simulation data for plotting and reporting.

| Variable / Function         | Description                    | Paper/Supplement Correlation | Implementation Detail                                                                                                          |
|:--------------------------- |:------------------------------ |:---------------------------- |:------------------------------------------------------------------------------------------------------------------------------ |
| `getalphasignaling`         | `getalphasignaling(popt, ...)` | **Code:** Analysis           | Calculates "Alpha" signals. Determines which signal is most prevalent for Type 0.                                              |
| `getalphastyped0`           | `getalphastyped0(...)`         | **Code:** Analysis           | Counts signal frequencies per dimension for Type 0.                                                                            |
| `execsruns`                 | `execsruns` array              | **Code:** Analysis           | Tracks executive attention patterns per run.                                                                                   |
| `oppsitiondisruns`          | `oppositiondisruns`            | **Code:** Analysis           | Counts opposition groups with maximal dissimilar signals.                                                                      |
| `agreeruns`                 | `agreeruns`                    | **Code:** Analysis           | Counts agreement groups with maximal dissimilar signals.                                                                       |
| `greetings_results`         | `greetings_results(...)`       | **Paper:** Section 2.3       | Aggregates final states. Calculates `simple_stat_000` (Success Rate).                                                          |
| `genBS_results`             | `genBS_results(...)`           | **Paper:** Section 2.3       | Aggregates final states. Calculates `simple_stat_002` (Coordination Matrix).                                                   |
| `genBSexec_results`         | `genBSexec_results(...)`       | **Code:** Analysis           | Aggregates final states for Exec version. Calculates `simple_stat_001` (Signal Matrix).                                        |
| `mutual_info`               | `mutual_info(...)`             | **Paper:** Section 4.2       | Calculates Shannon Mutual Information between signals and actions.                                                             |
| `average_mutual_info`       | `average_mutual_info(...)`     | **Code:** Analysis           | Averages MI across multiple runs.                                                                                              |
| `repconvert`                | `repconvert(...)`              | **Paper:** Section 2.2       | Converts integer profiles back to `sigurnsf`/`recurnsf` for analysis.                                                          |
| `execrepconvert`            | `execrepconvert(...)`          | **Code:** Analysis           | Converts integer profiles + execs back to urns.                                                                                |
| `execrepconvertRE`          | `execrepconvertRE(...)`        | **Code:** Analysis           | Converts RL urns back to profiles for analysis.                                                                                |
| `flip7zero_correction`      | `flip7zero_correction(...)`    | **Code:** Bug Fix            | Corrects "zero error" where normalization results in a count of 0 for a strategy that should exist. Resets to random profiles. |
| `time_update`               | `time_update(...)`             | **Paper:** Table 4 (Data)    | Aggregates `recurnsf` into `typed_time`. Calculates `typed_count_norm` to normalize agent influence.                           |
| `final_simple_stat_001`     | `final_simple_stat_001`        | **Code:** Output             | Stores the 3x3x3 matrix of coordination preferences.                                                                           |
| `final_simple_stat_002`     | `final_simple_stat_002`        | **Code:** Output             | Stores the 3x3x3 matrix of "Top A" counts.                                                                                     |
| `final_socialsig_aggregate` | `final_socialsig_aggregate`    | **Code:** Output             | Stores the aggregate signal frequencies over time.                                                                             |
| `final_action_aggregate`    | `final_action_aggregate`       | **Code:** Output             | Stores the aggregate action frequencies over time.                                                                             |
| `final_sigurns`             | `final_sigurns`                | **Code:** Output             | Final state of signal urns.                                                                                                    |
| `final_recurns`             | `final_recurns`                | **Code:** Output             | Final state of response urns.                                                                                                  |
| `final_typed_time`          | `final_typed_time`             | **Code:** Output             | Time-series data for typed actions.                                                                                            |
| `final_signal_time`         | `final_signal_time`            | **Code:** Output             | Time-series data for signal actions.                                                                                           |

---

### 7. Implementation Details Not Explicitly in Text

While the paper describes the *theory*, the code contains specific implementation details that deviate slightly or add complexity:

1. **`epsilonf` (Numerical Stability):**
   * **Code:** `csrand = (rngf.random()+epsilonf)*cscumsum[-1]`
   * **Detail:** The paper implies uniform sampling. The code adds `float_info.epsilon` to the random draw to ensure the cumulative sum is never exactly zero, preventing division by zero errors during probability sampling.
2. **`floor` (Discretization):**
   * **Code:** `stratdex = floor(stratdex)`, `agtype = floor(popt[popdex1])`
   * **Detail:** The paper uses continuous probabilities. The code converts continuous signal values (e.g., `sigmultipliers`) into discrete integer indices using `floor()` to index into numpy arrays.
3. **`cascade rounding` (Integer Counting):**
   * **Code:** `intround = np.around(floattotal)`, `incriment = intround-integertotal`
   * **Detail:** In `genBSreplicator_single_tstep`, the code rounds float counts to integers. It uses a "cascade" method to distribute rounding errors (increments) so the total population count remains exactly equal to `typestotal`. This prevents the population from drifting due to rounding errors.
4. **`execurnsf` (Executive Attention):**
   * **Code:** `execurnsf[idx1403][idx1400] = 1` (if attended)
   * **Detail:** The paper mentions "executives" in Section 4.1. The code implements this as a separate array (`execurnsf`) that tracks whether an agent is paying attention to a specific signaling dimension (1 = attend, 0 = ignore). This is distinct from the signal urns (`sigurnsf`).
5. **`profilecaps` (Urn Size):**
   * **Code:** `profilecaps[pcdex] = sigperdim[pcdex]`
   * **Detail:** In the RL simulation, `profilecaps` determines the *maximum* number of balls an urn can hold (or rather, the number of discrete options). It ensures that if an agent has 2 signal options, the urn has 2 slots for balls.
6. **`random2picks` (Action Selection):**
   * **Code:** `csdraw[cscumsum<csrand] = 1`
   * **Detail:** This is a cumulative probability sampling method. It converts the probability distribution in the urn (`csurn`) into a specific action (`csdraw`) for the timestep.
7. **`final_simple_stat_002` (Top A Count):**
   * **Code:** `if (np.argmax(...) == t0sigs_index[intdex7]) and ...`
   * **Detail:** This variable counts specific "Top A" outcomes where specific signal/action combinations are dominant. The logic is hardcoded to check for specific indices defined in `t0sigs_index` (e.g., `[[1, 0], [1, 0]]`).
8. **`genBSreplicator_full_play` vs `genBSreplicator_full_play_typed`:**
   * **Code:** `genBSreplicator_full_play` (Untyped), `genBSreplicator_full_play_typed` (Typed).
   * **Detail:** The "Typed" versions wrap the "Untyped" functions to convert numpy arrays into `numba.typed.List` for memory efficiency and to return them in a format compatible with the Jupyter notebook's data storage.
9. **`muterandoms10` (Mutation Logic):**
   * **Code:** `mutations10 = numbagreater(..., mutateratef)`
   * **Detail:** Uses `numbagreater` to apply mutation rates. `mutateratef` is a list `[0.01, 0.1]`. The first value is the probability an agent is selected for mutation. The second value is the probability a *bit* in their profile mutates.
10. **`repmutation_smart` (Smart Mutation):**
    * **Code:** `muteprofilestrings10 = numbagreater3d(...)`
    * **Detail:** A more complex mutation function that tracks specific bits (`profile_length`) to mutate, rather than just random integer replacement.

### 8. Mapping to "Single Embedding" vs "Multi Embedding"

* **Single Embedding Structure:**
  * **Code:** `genBS_f7_RothErevExec_full_play_typed` (Notebook 1) with `sigdimensions=2`.
  * **Paper:** Section 4.2.
  * **Logic:** The code uses `getalphasignaling` to determine which dimension is the "Alpha" signal. The `execassort_array_create` function modifies assortment to only count matches on non-null signals.
* **Multi Embedding Structure:**
  * **Code:** `genBSreplicatorexec_full_play_typed` (Notebook 2) with `sigdimensions=2`.
  * **Paper:** Section 4.4.
  * **Logic:** Uses `repmutation_smart` to handle profile strings of length `sigdimensions + numsignalsf`.

### 9. Summary of Variable Flow

1. **Setup:** Notebooks define `population_size`, `sigdimensions`, `coordination_preferences`.
2. **Init:** `replicator_initialize` or `RothErevExec_initialize` creates `sigurnsf`, `recurnsf`, and `profiles_indexed`.
3. **Loop:** `genBS_full_play` calls `greetings_check_success` or `genBS_check_success`.
4. **Learning:** `utilcompute` (Replicator) or `executilcomputeREforget` (RL) updates counts/urns.
5. **Mutation:** `muterandoms10` or `repmutation_smart` introduces noise.
6. **Assortment:** `assort_array_create` calculates who interacts with whom.
7. **Data:** `time_update` aggregates data for `final_simple_stat_001`.
8. **Output:** Notebooks plot `final_simple_stat_001` and calculate `average_mutual_info`.

Every variable in the code maps to a specific parameter or state variable defined in the Paper or Supplement, with the implementation details (like `floor`, `epsilon`, `cascade rounding`) added for numerical stability and computational efficiency.
