# The Definitive Code–Paper Mapping
## "The Evolution of Identity Signals for Coordination in Diverse Societies"

*This document provides a complete, bidirectional, error-checked mapping between every component
of the paper and supplement and every active subfunction, variable, and implementation detail
in the two simulation codebases. It is organized to serve both as a top-down explanation
(paper concept → code) and a bottom-up reference (code function → paper passage). Legacy and
inactive functions are catalogued separately to prevent misattribution.*

---

## Part 1: Notebook and Script Identification

### The Two-Script Architecture

Each notebook imports from its own dedicated Python script. They are **not** a shared codebase,
though both scripts contain many functionally identical analysis and assortment utility functions.

| Notebook | Python Script | Learning Dynamic |
|:---|:---|:---|
| `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb` | `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py` | **Discrete Replicator Dynamics** — Paper Section 2.2 |
| `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb` | `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py` | **Roth-Erev Reinforcement Learning with Forgetting** — Supplement Appendix B |

**Why this identification is unambiguous:** The replicator notebook's filename contains
`FLIP7_repEXEC` and the script name contains `SMARTmutation`, both indicating the replicator
variant with smart two-parameter mutation. The RL notebook's filename contains `BZforget3`
(referencing the forgetting mechanism) and `RothErevExec`, and its script name likewise
contains `BZforget3` and `RothErev`. The top-level callable imported from each script
confirms the identification: `genBSreplicatorexec_full_play_typed` (replicator) and
`genBS_f7_RothErevExec_full_play_typed` (RL).

Both notebooks target the **single-embedding structure** parameter space from Section 4.2
(Table 7 in the paper): three agent types (Lagashites, Girshites, Akkadians), two signaling
dimensions with one non-null signal per dimension, coordination preferences matching Table 7.

---

## Part 2: Shared Structural Foundations

The following components appear in both scripts and implement model elements common to both
learning dynamics. Where the implementations differ slightly between scripts, this is noted.

---

### 2.1 Population Array: `population_array`

**Paper connection — Section 2.2:** "We initialized a population of N = 1,000 agents with
a fixed distribution of preference types."

`population_array(population_sizef, percent_agents_per_typef)` produces a 1D integer array
`popt` of length `population_size` where `popt[i]` holds the preference type index (0, 1, or 2)
of agent `i`. Agents of the same type are stored contiguously, partitioned according to the
proportions in `percent_agents_per_typef`.

In the **replicator notebook**: `population_size = 1000`, matching Table 4 of the paper for
the Section 4.2 single-embedding simulations.

In the **RL notebook**: `population_size = 500`, matching Appendix B: "we ran simulations
for a population of 500 agents."

The proportions sweep over Lagashite offset β and Girshite offset β so that Lagashites =
0.33 + β, Girshites = 0.33 − β, Akkadians = 0.34. This implements the paper's β parameter
controlling the proportional imbalance between the two smaller Sumerian groups.

---

### 2.2 Strategy Profile Representation

**Paper connection — Sections 3.1 and 4.1:** "Agents' strategy profiles can be expressed as
strings of length 1 + θ, where θ is the number of potential signals that can be received."
Extended to multiple dimensions in Section 4.1: "strategy profiles are now strings of length
K + θ where θ is the number of possible signal combinations."

#### The Profile String

Each strategy profile is a vector of integer values of length `sigdimensions + numsignals`.

- **First `sigdimensions` positions** — the signal broadcast in each dimension. Value 0
  means the agent broadcasts the null signal (not attending). A value of 1 means the agent
  broadcasts the one non-null signal available in that dimension.

- **Next `numsignals` positions** — the action played when observing each of the
  `numsignals` possible combined incoming signals (indexed 0 to numsignals − 1). For the
  single-embedding sweep: numsignals = 2 × 2 = 4 combined signals, with 3 possible actions
  each (Lagash greeting, Girsu greeting, Akkadian greeting).

For the single-embedding sweep parameters (`sigdimensions = 2`, `numsignals_perdim = [2, 2]`,
`numactions = 3`): each profile string has length 6. There are 2 × 2 × 3⁴ = 324 possible
distinct strategy profiles (`numprofiles = 324`).

#### `repmultipliers` / `dimactions`

`dimactions` is the array of valid option counts at each profile position. For the
single-embedding sweep: `[2, 2, 3, 3, 3, 3]` — two signal options per dimension, three
action options per signal-response position.

`repmultipliers` is the mixed-radix positional weight vector computed from `dimactions` as a
shifted cumulative product: `repmultipliers[k]` is the stride for position `k` in the profile
integer encoding. Together, `repmultipliers` and `dimactions` implement the bijective mapping
between profile integer indices (0 to numprofiles − 1) and full profile string vectors. This is
the code's concrete realization of the paper's abstract "string" representation.

#### `profiles_indexed`

A 2D integer array of shape `(numprofiles, sigdimensions + numsignals)`. Row `k` holds the
full strategy string for profile index `k`. Built once during initialization by lexicographic
enumeration of all valid profile combinations. Used as an O(1) decode table: any function
holding an agent's integer profile index can retrieve the full signal/action string in constant
time via `profiles_indexed[idx]`.

#### `sigmultipliers`

A separate positional weight vector covering only the `sigdimensions` signal positions (not
the action positions). For `numsignals_perdim = [2, 2]`, `sigmultipliers = [1, 2]`. Used to
convert a multidimensional signal broadcast `[s1, s2]` into a single combined signal index:
`combined_sig = s1 * 1 + s2 * 2`. This combined signal index is what gets used to look up
an agent's receiver action and to index the assortment matrix.

#### `profilecaps`

An array of shape `(sigdimensions + numsignals,)` recording the number of valid options at
each profile position. Equivalent to `dimactions`. Used by `random2picks` in the RL model to
bound urn index sampling, and by mutation functions to constrain newly drawn profile elements
to valid ranges.

---

### 2.3 Assortment / Homophily

**Paper connection — Appendix A:** "A homophily parameter h can be included to model
contexts in which people interact more with those who broadcast the same social signals."
The assortment multiplier H(h, x, i) is defined as:

> H(h, x, i) = N / Σ_{j∈Y} [S(h, x, j)] × S(h, x, i)

where S(h, x, j) = (2 × d)^h if profile x and profile j broadcast the same signal in d
dimensions, and S(h, x, j) = 1 otherwise.

**Paper connection — Appendix B (RL extension):** "To represent assortment, we adjusted
agents' payoff by multiplying it by H(h, x, i)." The RL model uses the same formula,
generalized to multiple dimensions: S(h, x, j) = (2 × d)^h for d matching non-null
signal dimensions.

The assortment computation is split across three functions, all present and functionally
identical in both scripts.

#### `execassort_array_create`

Builds a static 2D array of shape `(numsignals, numsignals)` — the raw assortment weights
before normalization. For each pair of combined signal indices (i, j):

1. Decode i and j back to their per-dimension signal values using `sigmultipliers`.
2. Count `d` = the number of dimensions where both signals are **non-null** and equal.
   The "exec" prefix is critical: dimensions where either signal is null (0) are excluded
   from the match count entirely. This is what distinguishes `execassort_array_create` from
   the legacy `assort_array_create` (which counts all matching dimensions, including
   mutual-null ones, and is not called by either active notebook).
3. Set `assort_array[i][j] = base_connection_weights[d] ** homophily_factor`.

`base_connection_weights = [1, 2, 4, 8, 16, 32]`, so `base_connection_weights[d] = 2^d`.
With `homophily_factor = h`, the weight is `(2^d)^h = (2d)^h`, implementing S(h, x, j)
from the paper exactly.

This array is computed **once before the main simulation loop** and reused each timestep,
since the signal option structure never changes.

**Notebook parameters:**
- Replicator notebook: `homophily_factor = 1` — full assortment (same-signal agents interact
  at twice the baseline rate per shared non-null dimension).
- RL notebook: `homophily_factor = 0` — random mixing (`base_connection_weights[d]^0 = 1`
  for all d, so all signal pairs receive equal weight).

#### `get_sig_counts` (replicator) / `get_sig_countsRE` (RL)

Computes the population's current signal distribution: how many agents are broadcasting
each combined signal index. This provides the denominator term Σ_{j∈Y} [S(h, x, j)] for
the H(h, x, i) normalization.

- **`get_sig_counts`** (replicator): iterates over all profile indices in
  `profile_count_untyped` to tally signal frequencies. Works from profile count arrays.
- **`get_sig_countsRE`** (RL): iterates over `agentprofilepicks` (the per-agent profile
  draws for the current timestep). Works from the freshly drawn profiles rather than
  persistent count arrays, since the RL model does not maintain running profile counts.

#### `find_assort_multipliers`

Applies the normalization factor to produce the final H(h, x, i) matrix. For each row
(signal i), computes:

`assort_multipliers[i][j] = assort_array[i][j] × N / Σ_k (assort_array[i][k] × assort_sig_counts[k])`

This is the full Appendix A formula. The resulting matrix entry `assort_multipliers[i][j]`
is the factor by which payoffs are scaled when an agent broadcasting combined signal i
interacts with an agent broadcasting combined signal j. When h = 0, all entries equal 1
(random mixing). When h = 1 in a one-dimensional system, pairs sharing the signal get
weight 2 and pairs differing get weight ≈ 1 (depending on population composition), matching
the paper's claim that "agents who broadcast the same signal are twice as likely to interact."

Called at the **start of every timestep** because signal counts change as the population
evolves.

---

### 2.4 The Null-Signal Interaction Rule

**Paper connection — Section 3.3:** "When an agent does not attend to signals (i.e., when
she broadcasts 0), she interacts with all other agents as if they had also broadcast 0. In
other words, she ignores all signals, so that actions cannot be chosen based on the social
signal that was broadcast."

Implemented inside both `executilcompute` (replicator) and `executilcomputeREforget` (RL).
Before computing which action each agent takes in response to the other's signal, the code
applies dimension-wise masking:

```
for each signal dimension d:
    if A's profile[d] == 0 OR B's profile[d] == 0:
        set both A's effective signal[d] = 0
        set both B's effective signal[d] = 0
```

The **effective** combined signal used to look up each agent's conditional action is computed
from the masked profile. An agent broadcasting null in all dimensions will always look up
their action for combined signal index 0, meaning they play the same action regardless of
what anyone broadcasts — implementing "ignoring all signals."

Crucially, the **assortment multiplier** is computed from agents' **actual** (pre-masking)
broadcast signals, not the masked effective signals. Homophily depends on what signals are
physically broadcast; the null-masking rule only governs which urn position an agent uses
to choose their action.

---

### 2.5 Signal Cost

**Paper connection — Section 3.3:** "a signal cost c that an agent incurs if she broadcasts
any signal other than 0. When an agent uses a nonzero signal, its payoff from an interaction
is decreased by c."

**Extension to multiple dimensions — Section 4.1:** "we assume the signal cost c is incurred
for each dimension attended to."

In the code:
- `sigcostf` is set to a **negative** number (e.g., −0.0005 in the replicator notebook,
  −0.000125 in the RL notebook).
- Inside the utility / reinforcement loop, `totalAsigcost = (number of non-null signal
  dimensions in profile A) × sigcostf`.
- `totalAsigcost` is **added** to the payoff value (since it is negative, this reduces
  payoffs for attending agents).
- The cost is applied on **every** interaction — both successes and failures — and is
  proportional to how many dimensions the agent actively signals in. An agent attending to
  both dimensions pays 2c per interaction.

---

### 2.6 Coordination Preferences and Failure Payoff

**Paper connection — Tables 5–10 and Appendix B Table 1.**

`coordination_preferencesf` is a **2D array of shape [numtypes, numactions]**. Entry
`[k][a]` gives the payoff that preference type k receives when coordinating on action a.
For the single-embedding sweep, this implements Table 7: Lagashites most prefer the Lagash
greeting (payoff 1), have equal preference for Girsu greeting (payoff 0) and accept the
Akkadian greeting (payoff 0.5); Girshites most prefer the Girsu greeting (payoff 1 + α)
and also like the Lagash greeting (payoff 1); Akkadians only care about the Akkadian greeting.

`genBSpunishf` is the payoff received when coordination **fails**. Set to 0 in both active
notebooks, matching the generalized BoS structure where failure to coordinate yields zero.

In the **RL notebook**, the coordination preferences are divided by 200:
`coordination_preferences = np.array([[1.5, 0, .85], [1.25, 1.5+α, .85], [0, 0, 1]]) / 200`.
This brings payoff magnitudes to the range 0.005–0.0075, matching Appendix B Table 1 exactly
(e.g., Lagashites' Lagash-greeting payoff = 1.5/200 = 0.0075). The 1/200 scaling keeps
reinforcement increments small relative to initial urn weights of 1, preventing early
interactions from irreversibly locking in urn distributions before agents have had time to
explore.

---

### 2.7 Output and Analysis Functions

These functions are called post-simulation in both notebooks and have identical implementations
in both scripts.

#### `genBSexec_results` (replicator) / `genBS_resultsRE` (RL)

Computes four output arrays from the current population state:

- **`socialsig_aggregate[type][signal]`** — for each preference type, counts agents whose
  modal signal (argmax of signal urn) equals each combined signal index.
- **`action_aggregate[type][signal][action]`** — for each type, counts how often each
  action is taken in response to each incoming combined signal.
- **`simple_stat_001[y_type][x_type][action]`** — the argmax action: the most common action
  that agents of type y take when observing the modal signal of type x. This is a "soft"
  summary that accepts any dominant pattern.
- **`simple_stat_002[y_type][x_type][action]`** — the strict version: records the action
  only if (a) >75% of type x agents share the same modal signal AND (b) >75% of type y
  agents take the same action in response to that signal. This is the primary outcome
  classification variable used for the sweep heatmaps.

`simple_stat_002`'s 75% threshold implements the paper's optimality criteria in a computable
form: it identifies runs where enough coordination on a clear signal-action mapping has
emerged to count as a "clean" behavioral equilibrium.

#### `time_update`

**Paper connection — Table 4:** Records time-series snapshots at every `record_interval`
timestep. Aggregates per-agent `recurnsf` values (weighted by agent signal) into four 4D
arrays: `typed_time`, `signal_time`, `typed_time_norm`, `signal_time_norm`. These track
the evolution of coordination behavior over time for single-run trajectory visualization.
Not used in cross-run sweep statistics.

#### `getalphasignaling` and `getalphastyped0`

A post-processing function essential for cross-run aggregation. The labeling of which signal
is "signal 1" versus "signal 2" in a dimension is arbitrary — different runs may converge
to the same coordination outcome with signals relabeled. `getalphasignaling` standardizes
across runs by:

1. Identifying type 0's (Lagashites') most common signal in each dimension as the "alpha"
   signal for that run.
2. Relabeling all signal indices across that run so type 0's modal signal is consistently
   called "alpha."
3. Computing `execsruns` — which dimensions each type most commonly attends to (inferred
   from argmax of signal urns).
4. Computing `oppositiondisruns` — runs where the modal signal of the "opposition" type
   (type 1, Girshites) is maximally different from the alpha signal.
5. Computing `agreeruns` — runs where the modal signal of the "agree" type (type 2,
   Akkadians) matches the alpha signal in at least one dimension.
6. Computing `multidimsignalscount0` — frequency of each signal combination per type.

Without this relabeling, averaging signal distributions across runs would conflate
equivalent coordination equilibria that happen to have used different signal labels.

#### `mutual_info` and `average_mutual_info`

**Paper connection — Section 4.3:** "Using partial information decomposition (Williams and
Beer, 2010) and focusing on Varley and Kaminski's definition of synergistic information, we
found that, for β = 0, the synergistic information in the intersectional equilibrium was
around 49% of the total information."

`mutual_info` computes the Shannon mutual information I(S; A) between the combined signal
broadcast by an agent and the action taken in response, across the population. Used to
quantify how much information the signaling system carries about coordination behavior.
`average_mutual_info` averages this across all simulation runs. These functions support
the synergistic information analyses of Section 4.3, though base mutual information is a
building block rather than synergistic information itself.

---

### 2.8 Convergence / Early Termination

**Paper connection — Table 4, footnote:** "every 100 timesteps it was checked whether the
distribution of agents' strategy profiles was unchanged. If so, the simulation was halted
prior to reaching 4 × 10^4 timesteps."

**Replicator:** Checks every **100** timesteps by comparing `agentprofilesCOPY` to
`agentprofiles` using exact integer-array equality. If the full per-agent profile assignment
is unchanged, the simulation has reached a stable absorbing state and halts.

**RL:** Checks every **1,000** timesteps by comparing `agentprofilepicksCOPY` to
`agentprofilepicks`. The interval is 10× longer because the stochastic urn sampling process
(`random2picks`) will produce different `agentprofilepicks` on successive timesteps even when
the underlying urn distributions are fully stable. The exact-equality check is therefore only
informative at longer intervals after which genuine stability can be inferred.

---

## Part 3: Replicator Dynamics — Complete Mapping

**Notebook:** `genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`
**Script:** `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`
**Paper reference:** Section 2.2, with extensions from Sections 3.3 and 4.1

The replicator notebook's main simulation call is:
`genBSreplicatorexec_full_play_typed(...)` → `genBSreplicatorexec_full_play(...)`

---

### 3.1 Initialization

#### `replicatorexec_initialize`

Sets up the structural arrays needed for the simulation:
- Builds `profiles_indexed` by lexicographic enumeration of all `numprofiles` strategy
  combinations.
- Randomly assigns each agent a one-hot signal per dimension (in `sigurnsf`) and a
  one-hot action per signal-response position (in `recurnsf`), establishing the urn-format
  baseline used for reporting.
- Initializes `execurnsf` to zeros (populated only at reporting time).
- Computes `sigperdimf` (signals per dimension) and `profilecaps`.

#### `genBSreplicatorexec_k_step` — **The Active Profile Initializer**

**Paper connection — Section 2.2:** "randomly assigned a behavior strategy to each."

This function performs the actual random initialization of strategy profiles for the
replicator model. It draws a uniformly random integer profile index from `[0, numprofiles)`
for each agent and fills:
- `agentprofiles[i]` — the profile index for agent i.
- `profile_count_typed[k][x]` — how many agents of type k hold profile x.
- `profile_count_untyped[x]` — total agents holding profile x.
- `typestotal[k]` — total agents of type k.

*Note: `genBSreplicator_first_tstep` is an alternative initializer present in the script
but **commented out** in the active code. `genBSreplicatorexec_k_step` is the active
initializer for all single-embedding replicator simulations.*

---

### 3.2 Utility Computation: `executilcompute`

**Paper connection — Section 2.2:** The utility function U_k(x) = Σ_{i∈Y} [M(i) × p_{xi,k}]
if x ≠ i, Σ_{i∈Y} [(M(i) − 1) × p_{xi,k}] if x = i. With assortment (Appendix A):
U(x) = Σ_{i∈Y} [H(h, x, i) × M(i) × p_{xi}] (excluding self-pairing with M(i) − 1).

`executilcompute` is the core utility calculation. For every ordered pair of distinct profiles
(A, B) currently present in the population:

1. **Compute assortment signal indices** — `assortsigA` and `assortsigB` from the first
   `sigdimensions` entries of each profile using `sigmultipliers`. These pre-masking signal
   indices are used for assortment lookup.

2. **Apply null-signal masking** — zero out any dimension where either profile has a 0,
   producing the effective signals `Asignal` and `Bsignal`. These determine which action
   each agent plays.

3. **Look up actions** — `Aaction = profiles_indexed[A][sigdimensions + Bsignal]` (the
   action A takes when observing B's effective signal), and symmetrically for B.

4. **Check coordination** — if `Aaction == Baction`, both agents coordinate. If not, they
   fail.

5. **Compute assortment multiplier** — `hfmultiplier = assort_multipliers[assortsigA][assortsigB]`.

6. **Compute signal cost** — `totalAsigcost = count_nonzero_dims(A_profile) × sigcostf`.

7. **Accumulate utility** — for each preference type k:
   - Success: `profile_utility_bytype[k][A] += hfmultiplier × (profile_count_untyped[B] or B−1 if A==B) × (coordination_preferencesf[k][Aaction] + totalAsigcost)`
   - Failure: replace `coordination_preferencesf[k][Aaction]` with `genBSpunishf` (= 0).

The result `profile_utility_bytype[k][x]` is U_k(x) — the total expected payoff of strategy
profile x for a preference-type-k agent, summed across all possible interactions in the
current population weighted by assortment.

---

### 3.3 Replicator Update: `genBSreplicator_single_tstep`

**Paper connection — Section 2.2:** N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) × [U_k(x) − Avg(U_k(i))_{i∈X}]. "After adjusting the prevalence of strategy profiles, their quantity is re-normalized so that the number of agents of each preference type remains constant throughout a simulation."

For each preference type k:

1. **Compute average utility** — `typesaverage[k] = mean(profile_utility_bytype[k][x])` over
   all profiles x currently present for type k (those with `profile_count_typed[k][x] > 0`).

2. **Apply replicator equation** — `profile_count_typed[k][x] += profile_count_typed[k][x] × (profile_utility_bytype[k][x] − typesaverage[k])` for all x.

3. **Non-negativity clamp** — any `profile_count_typed[k][x] < 0` is set to 0. (Negative
   counts can arise when a profile's utility falls far below the type average; this is a
   discretization artifact of the deterministic equation.)

4. **Renormalization** — `profile_count_typed[k] *= typestotal[k] / sum(profile_count_typed[k])`.
   This preserves each type's total agent count exactly, implementing the paper's "re-normalized"
   step.

5. **Cascade integer rounding** — converts floating-point profile counts back to integers
   while preserving the total. Uses cumulative floating-point accumulation (`floattotal`) and
   `np.around`: as the cumulative float total crosses each integer threshold, one agent is
   assigned to that profile. This prevents population drift from repeated rounding.

6. **Rebuild `agentprofiles`** — produces the ordered per-agent profile index array from the
   updated integer counts, laying agents out contiguously by profile and type.

7. **Update `profile_count_untyped`** — `profile_count_untyped[x] = Σ_k profile_count_typed[k][x]`.

#### `flip7zero_correction`

A numerical safety function called within `genBSreplicator_single_tstep` when cascade
rounding drives a preference type's total agent count to zero. This extremely rare event
(noted in code comments as roughly 1-in-10,000) is corrected by randomly reinitializing that
type's profile distribution. Not described in the paper; a necessary implementation safeguard.

---

### 3.4 Mutation: `muterandoms10_smart` and `repmutation_smart`

**Paper connection — Section 2.2:** "We also allow for the possibility that an agent's
strategy profile is not copied faithfully, or 'mutates.' This is governed by two parameters,
m0 and m1. Each agent is selected for mutation with probability m0. If selected, each element
in the string expression of the agent's strategy profile is assigned a random value from
among the allowable values with probability m1."

Mutation is applied every 10 timesteps (the check `if idxN0 % 10 == 0` in the main loop).

#### `muterandoms10_smart`

Pre-generates 10 rounds of mutation data at once to avoid the overhead of calling the random
number generator inside the tight numba loop:

- `mutations10[t][i]` — a Bernoulli draw with success probability m0 for each agent i at
  each relative timestep t. Marks whether agent i is selected for mutation.
- `newprofiles10[t][k]` — a pre-drawn target profile index for the k-th mutation event at
  timestep t. The profile is drawn uniformly from all `numprofiles` possibilities.
- `muteprofilestrings10[t][k][pos]` — a Bernoulli draw with success probability m1 for each
  position `pos` in the profile string, for the k-th mutation event at timestep t. Marks
  which positions actually change.

This batch of 10 rounds is regenerated every 10 timesteps.

#### `repmutation_smart`

Applies the pre-drawn mutation data to the current population:

For each selected agent (those where `mutations10[t][i] == 1`):
- A new profile is constructed position-by-position: position `pos` takes the value from
  `newprofiles10` if `muteprofilestrings10[...][pos] == 1` (probability m1), or retains the
  agent's old value otherwise.
- The new profile index is computed from the resulting string using `repmultipliers`.
- `profile_count_typed[k][old_profile]` is decremented; `profile_count_typed[k][new_profile]`
  is incremented.
- `profile_count_untyped` is updated correspondingly.

This implements the paper's two-parameter mutation exactly: m0 governs whether an agent's
profile changes at all; m1 governs which positions within the profile change. The "smart"
suffix distinguishes these from legacy functions (`muterandoms10`, `repmutation` without
`_smart`) that use only m0 and replace the entire profile rather than individual positions.

---

### 3.5 Profile-to-Urn Conversion: `execrepconvert`

After the simulation reaches a recording interval, `execrepconvert` converts the replicator's
compact integer-profile representation back to the urn-format arrays expected by reporting
functions:

- For each agent i with profile index `agentprofiles[i]`:
  - Sets `sigurnsf[d][i][profiles_indexed[agentprofiles[i]][d]] = 1` for each dimension d.
  - Sets `recurnsf[i][s][profiles_indexed[agentprofiles[i]][sigdimensions + s]] = 1` for each signal combination s.
  - Sets `execurnsf[i][d] = 1` if agent i's profile has a non-null signal in dimension d, else 0.

The `execurnsf` array is populated **only here** — it is not a dynamic tracking variable
during the simulation, but a reporting-time inference from the current profile.

---

### 3.6 Full Simulation Loop: `genBSreplicatorexec_full_play`

The main loop structure per timestep:

```
1. get_sig_counts(...)           → assort_sig_counts
2. find_assort_multipliers(...)  → assort_multipliers
3. executilcompute(...)          → profile_utility_bytype
4. genBSreplicator_single_tstep(...)  updates profile_count_typed/untyped, agentprofiles
5. if idxN0 % 10 == 0:
       muterandoms10_smart(...)   → mutations10, newprofiles10, muteprofilestrings10
   repmutation_smart(...)        updates profile_count_typed/untyped
6. if (idxN0 + 1) % record_interval == 0:
       execrepconvert(...)        → sigurnsf, recurnsf, execurnsf
       genBSexec_results(...)     → socialsig_aggregate, action_aggregate, simple_stat_001/002
       time_update(...)           → typed_time, signal_time
7. if idxN0 % 100 == 0:
       convergence check: if agentprofilesCOPY == agentprofiles → halt
```

---

## Part 4: Roth-Erev Reinforcement Learning — Complete Mapping

**Notebook:** `genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`
**Script:** `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`
**Paper reference:** Supplement Appendix B

The RL notebook's main simulation call is:
`genBS_f7_RothErevExec_full_play_typed(...)` → `genBS_f7_RothErevExec_full_play(...)`

---

### 4.1 Initialization

#### Urn Initialization — Done in the Notebook

**Paper connection — Appendix B:** "Our implementation of reinforcement learning with
forgetting can be thought of as each agent in the population having an urn for each location
in the string representation of their strategy profile. Each urn begins a simulation with
one ball corresponding to each possible action for that location in the strategy profile string."

The urn contents are initialized to `inertia = 1` **directly in the notebook** before the
simulation call:

```python
sigurns = [np.ones([population_size, numsignals_perdim[d]]) * inertia
           for d in range(sigdimensions)]
recurns = np.ones([population_size, numsignals, numactions]) * inertia
```

This sets every urn entry to 1 — one ball per option — implementing the paper's initialization
exactly. `sigurns[d][i][s]` = weight for agent i broadcasting signal s in dimension d.
`recurns[i][sig_combo][action]` = weight for agent i taking `action` given incoming combined
signal `sig_combo`.

#### `RothErevExec_initialize`

Computes structural parameters **only** — does not touch urn contents:
- `sigperdimf[d]` = number of signal options in dimension d.
- `profilecaps` = valid option count at each profile position (used by `random2picks` to
  bound draws).

This function also computes `profiles_indexed` and `repmultipliers` for consistency with the
replicator initialization, though these arrays are used primarily for reporting in the RL model.

---

### 4.2 Per-Timestep Strategy Drawing: `random2picks`

**Paper connection — Appendix B:** "In a single timestep, an agent begins by drawing one
ball from each of their urns at random with equal probability. These draws then determine
the strategy profile that the agent employs for the entire timestep while interacting with
every other agent in the population."

`random2picks` samples one profile per agent from their current urns and stores the result
in `agentprofilepicks[i][pos]`. For each agent i:

- For each signal dimension d: draw one signal value from `sigurns[d][i]` proportionally
  to its weights using the cumulative-sum threshold method. Store in `agentprofilepicks[i][d]`.
- For each combined signal combo s (0 to numsignals − 1): draw one action from `recurns[i][s]`
  proportionally to its weights. Store in `agentprofilepicks[i][sigdimensions + s]`.

The cumulative-sum threshold sampling works as follows: a random value `csrand` is drawn from
`(0, 1]` (the ε offset prevents exactly-zero draws); the cumulative sum of urn weights is
computed; the drawn option is the index where `cscumsum >= csrand` first. This is a correct
implementation of weighted random sampling proportional to urn counts.

`agentprofilepicks` thus holds each agent's complete drawn strategy — their broadcast signal
in each dimension and their intended action for every possible incoming signal — for the entire
timestep. It is overwritten at the start of every subsequent timestep.

**Performance detail:** `picks10` is a pre-generated 3D array of random draws for 10 timesteps,
refreshed every 10 steps (`if idxN0 % 10 == 0`). `random2picks` uses these pre-generated
numbers rather than calling the RNG inside the numba JIT loop for each individual agent-draw,
reducing overhead significantly.

---

### 4.3 RL Update: `executilcomputeREforget`

**Paper connection — Appendix B:** "For each individual that is interacted with, if an agent
receives a non-zero payoff the urns that contributed to their actions are reinforced.
Reinforcements can be thought of as adding additional balls of the type that was drawn to the
urn that it was drawn from." / "at the end of a timestep, forgetting is implemented by
multiplying the number of balls of each type by a forgetting factor 0 < φ < 1. For any ball
type whose quantity falls below 1 in any urn, that quantity is raised to 1; this helps
preserve the possibility of that action being chosen in the future."

**Paper connection — Appendix B assortment:** "To represent assortment, we adjusted agents'
payoff by multiplying it by H(h, x, i)."

This is the core RL update function, replacing both `executilcompute` and
`genBSreplicator_single_tstep` from the replicator path. It proceeds in three phases:

#### Phase 1: Forgetting (applied first, before reinforcement)

```
for each signal dimension d, each agent i, each signal value s:
    sigurns[d][i][s] *= forgetf

for each agent i, each signal combo sig, each action a:
    recurns[i][sig][a] *= forgetf
```

All urn weights are multiplied by `forgetf` = 0.998 at the **beginning** of the function,
before any interactions are processed. This ordering — forgetting before reinforcement — is
not stated explicitly in Appendix B but is what the code implements. The effect is that old
payoff history decays slightly before the current timestep's reinforcement is added.

#### Phase 2: Reinforcement (selective urn update)

For every ordered pair of agents (A, B) — the loop structure ensures each unordered pair is
processed once, with both agents' urns updated from that single pass:

1. **Read profiles** from `agentprofilepicks` (the draws made at the start of this timestep).

2. **Apply null-signal masking** — same logic as in `executilcompute`: zero out any dimension
   where either agent's drawn signal is 0, producing effective signals `Asignal` and `Bsignal`.

3. **Look up actions** — `Aaction = agentprofilepicks[A][sigdimensions + Bsignal]`.

4. **Compute assortment multiplier** — `Ahfmultiplier = assort_multipliers[assortsigA][assortsigB]`
   where `assortsigA` is computed from A's actual (pre-masking) signal broadcast.

5. **Compute signal cost** — `totalAsigcost = count_nonzero_dims(A_profile) × sigcostf`.

6. **Compute reinforcement** — if A and B coordinate (same action):
   `Areinforcement = Ahfmultiplier × (coordination_preferencesf[Atype][Aaction] + totalAsigcost)`
   If they fail to coordinate:
   `Areinforcement = Ahfmultiplier × (genBSpunishf + totalAsigcost)` (= Ahfmultiplier × sigcost since genBSpunishf = 0)

7. **Update only the urns that contributed to A's action this timestep:**
   - For each signal dimension d where A broadcast a non-null signal:
     `sigurns[d][A][agentprofilepicks[A][d]] += Areinforcement`
   - `recurns[A][Bsignal][Aaction] += Areinforcement`
   - Urns for dimensions A did not attend to, and receiver urns for signals A did not
     encounter this timestep, are left unchanged — implementing the paper's "only the urns
     that contributed to their actions are reinforced."

8. Symmetrically update B's urns.

**The all-pairs structure:** Rather than drawing random pairs, `executilcomputeREforget`
iterates over all agent pairs via modular index arithmetic. Each agent effectively interacts
with every other agent in the population each timestep. This is consistent with Appendix B's
tractability note and with the replicator's "large representative sample" assumption.

#### Phase 3: Floor Enforcement

After all pairs have been processed:
```
for each urn entry: if entry < 1: set entry = 1
```

Any urn entry that has been reduced below 1 (by forgetting alone or forgetting plus
reinforcement) is raised back to 1, implementing Appendix B's "that quantity is raised to 1;
this helps preserve the possibility of that action being chosen in the future."

---

### 4.4 Modal Profile Extraction: `execrepconvertRE`

Converts the continuous urn representation to a point estimate of each agent's current
"effective strategy" for reporting purposes. For each agent:
- The modal signal in each dimension = `argmax(sigurns[d][i])`.
- `execurnsf[i][d] = 1` if modal signal is non-null, else 0.

This is called **only at reporting intervals**, not during dynamics. `execurnsf` is a
reporting-time inference, not a dynamic simulation variable.

---

### 4.5 Mutation — Absent from the Active RL Loop

`mutateratef` is passed as a parameter to `genBS_f7_RothErevExec_full_play` but is **not
used** inside that function. No mutation calls appear in the active RL main loop. This is
intentional: exploration is provided by two other mechanisms:

1. **Stochastic urn sampling** via `random2picks` — even at a stable urn equilibrium, the
   drawn profile will vary across timesteps, maintaining a degree of behavioral diversity.

2. **The floor constraint** — keeping every urn entry at minimum 1 ensures that no action
   is ever permanently eliminated from an agent's repertoire, regardless of how seldom it
   has been reinforced.

---

### 4.6 Full Simulation Loop: `genBS_f7_RothErevExec_full_play`

The main loop structure per timestep:

```
1. random2picks(...)              → agentprofilepicks
   (using pre-generated picks10; refresh picks10 if idxN0 % 10 == 0)
2. get_sig_countsRE(...)          → assort_sig_counts (from agentprofilepicks)
3. find_assort_multipliers(...)   → assort_multipliers
4. executilcomputeREforget(...)   updates sigurns, recurns (forgetting → reinforce → floor)
5. if (idxN0 + 1) % record_interval == 0:
       execrepconvertRE(...)       → execurnsf (from urn argmax)
       genBS_resultsRE(...)        → socialsig_aggregate, action_aggregate, simple_stat_001/002
       time_update(...)            → typed_time, signal_time
6. if idxN0 % 1000 == 0:
       convergence check: if agentprofilepicksCOPY == agentprofilepicks → halt
```

---

## Part 5: Variable Reference Tables

### 5.1 Simulation Control Parameters (Set in Notebooks)

| Variable | Replicator Value | RL Value | Paper Reference |
|:---|:---|:---|:---|
| `population_size` | 1000 | 500 | Table 4; Appendix B |
| `numtypes` | 3 | 3 | Sec 3.3/4.2: Lagashites, Girshites, Akkadians |
| `sigdimensions` | 2 | 2 | Sec 4.1: K=2 signaling dimensions |
| `numsignals_perdim` | [2, 2] | [2, 2] | Sec 4.1: one non-null signal per dimension |
| `numsignals` | 4 (= 2×2) | 4 | Sec 3.1: total combined signal combinations |
| `numactions` | 3 | 3 | Tables 5–10: three greetings |
| `numprofiles` | 324 (= 4 × 3⁴) | 324 | Implementation: all valid strategy strings |
| `homophily_factor` | 1 | 0 | Appendix A: h parameter |
| `signal_cost` | −0.0005 | −0.000125 | Sec 3.3/4.1: cost c per active dimension |
| `genBSpunish` | 0 | 0 | Sec 2.2: failure payoff = 0 |
| `BZforget_multiplier` | N/A | 0.998 | Appendix B: φ = 0.998 |
| `inertia` | N/A | 1 | Appendix B: "one ball per action per urn" |
| `mutation_rate` | [0.01, 0.1] | passed but unused | Sec 2.2: [m0, m1] |
| `runlength` | 4×10^4 | 4×10^4 | Table 4 |
| `epsilon` | float_info.epsilon | float_info.epsilon | Implementation: prevents exact-zero sampling |

### 5.2 Strategy Representation Variables

| Variable | Shape | Role | Paper Reference |
|:---|:---|:---|:---|
| `profiles_indexed` | [numprofiles, sigdimensions+numsignals] | Lookup table: integer → full profile string | Sec 3.1, 4.1: strategy profile strings |
| `repmultipliers` | [sigdimensions+numsignals] | Mixed-radix weights: profile string → integer index | Implementation of string encoding |
| `sigmultipliers` | [sigdimensions] | Signal-only weights: signal vector → combined signal index | Implementation of combined signal |
| `dimactions` | [sigdimensions+numsignals] | Valid option count per profile position | Bounds for profile enumeration and mutation |
| `profilecaps` | [sigdimensions+numsignals] | Same as dimactions; used in RL urn size setup | Implementation |
| `base_connection_weights` | [33] | `[1, 2, 4, 8, 16, 32, ...]` — entry d = 2^d | Appendix A/B: S(h,x,j) = (2d)^h |

### 5.3 Population State Variables

| Variable | Model | Shape | Paper Reference | Notes |
|:---|:---|:---|:---|:---|
| `popt` | Both | [N] int | Sec 2.2: fixed preference type per agent | Produced by `population_array` |
| `typestotal` | Both | [numtypes] int | Sec 2.2: constant type sizes | Used for replicator renormalization |
| `profile_count_typed` | Replicator | [numtypes, numprofiles] float | Sec 2.2: N_{k,t}(x) | Core replicator state; float for arithmetic |
| `profile_count_untyped` | Replicator | [numprofiles] float | Sec 2.2: M(x) = Σ_k N_{k,t}(x) | Aggregated count; used as M(i) in utility |
| `agentprofiles` | Replicator | [N] int | Sec 2.2: agent's current strategy | Per-agent profile index; rebuilt each step |
| `profile_utility_bytype` | Replicator | [numtypes, numprofiles] float | Sec 2.2: U_k(x) | Utility matrix from `executilcompute` |
| `typesaverage` | Replicator | [numtypes] float | Sec 2.2: Avg(U_k) | Per-type mean utility for replicator update |
| `sigurns` | Both | List of [N, sig_per_dim] | Appendix B: sender urns | Replicator: one-hot; RL: continuous weights |
| `recurns` | Both | [N, numsignals, numactions] | Appendix B: receiver urns | Replicator: one-hot; RL: continuous weights |
| `execurnsf` | Both | [N, sigdimensions] int | Sec 3.3: attention flags | **Reporting-time only**; inferred from urn argmax |
| `agentprofilepicks` | RL only | [N, sigdimensions+numsignals] int | Appendix B: drawn strategy per timestep | Produced by `random2picks`; overwritten each step |
| `picks10` | RL only | [10, N, profile_length] float | Implementation | Pre-generated random draws for `random2picks` |

### 5.4 Assortment Variables

| Variable | Shape | Role | Paper Reference |
|:---|:---|:---|:---|
| `assort_array` | [numsignals, numsignals] | Raw assortment weights (before normalization) | Appendix A: S(h,x,j) = (2^d)^h |
| `assort_sig_counts` | [numsignals] | Current signal distribution | Appendix A: Σ_j denominator |
| `assort_multipliers` | [numsignals, numsignals] | Normalized H(h,x,i) | Appendix A: full H formula |
| `assortsigA` / `assortsigB` | scalar int | Combined signal index (pre-masking) | Appendix A: what is physically broadcast |
| `Asignal` / `Bsignal` | scalar int | Effective combined signal (post-masking) | Sec 3.3: what each agent uses for action lookup |
| `Aaction` / `Baction` | scalar int | Contingent action from profile | Sec 3.1: action conditioned on partner's signal |
| `hfmultiplier` / `Ahfmultiplier` | scalar float | H(h,x,i) for this pair | Appendix A/B: payoff scaling factor |

### 5.5 Payoff and Reinforcement Variables

| Variable | Model | Role | Paper Reference |
|:---|:---|:---|:---|
| `coordination_preferencesf` | Both | [numtypes, numactions] 2D array of payoffs | Tables 5–10; Appendix B Table 1 |
| `genBSpunishf` | Both | Scalar; failure payoff (= 0) | Sec 2.2: coordination failure yields 0 |
| `sigcostf` | Both | Scalar; negative signal cost per active dim | Sec 3.3, 4.1: cost c |
| `totalAsigcost` | Both | count_nonzero_dims × sigcostf; computed per interaction | Sec 4.1: total cost for attending dimensions |
| `Areinforcement` | RL | Ahfmultiplier × (payoff + totalAsigcost) | Appendix B: "payoff-many balls" |
| `forgetf` | RL | 0.998; φ forgetting factor | Appendix B: "multiply by φ" |

### 5.6 Mutation Variables (Replicator Only)

| Variable | Shape | Role | Paper Reference |
|:---|:---|:---|:---|
| `mutations10` | [10, N] bool | Agent-level mutation selection (m0) | Sec 2.2: probability m0 |
| `newprofiles10` | [10, max_mutations] int | Target profile indices for mutations | Sec 2.2: replacement profile |
| `muteprofilestrings10` | [10, max_mutations, profile_length] bool | Per-position mutation selection (m1) | Sec 2.2: probability m1 per element |

### 5.7 Output Variables

| Variable | Shape | Role | Paper Reference |
|:---|:---|:---|:---|
| `socialsig_aggregate` | [numtypes, numsignals] | Modal signal count per type | Sec 3.4: signal usage by type |
| `action_aggregate` | [numtypes, numsignals, numactions] | Action distribution per type per signal | Sec 3.2: conditional actions |
| `simple_stat_001` | [numtypes, numtypes, numactions] | Modal action (argmax) for each type-pair | Sec 4.2: coordination outcome detection |
| `simple_stat_002` | [numtypes, numtypes, numactions] | Strict modal action (>75% threshold) | Sec 4.2: optimality criteria |
| `typed_time` etc. | 4D arrays | Time-series of action/signal frequencies | Table 4: trajectory recording |
| `execsruns` | [runs, numtypes, sigdimensions] | Dimensions attended per type per run | Sec 3.3, 4.1: executive attention |
| `multidimsignalscount0` | per run | Signal frequency per type | Sec 4.2: signal distribution |
| `oppositiondisruns` | [runs] bool | Type 1's signal maximally differs from alpha | Sec 4.2/4.3: group structure analysis |
| `agreeruns` | [runs] bool | Type 2's signal matches alpha | Sec 4.2: group structure analysis |

---

## Part 6: Paper-to-Code Quick Reference

| Paper / Supplement Element | Code Implementation |
|:---|:---|
| **N** (population size) | `population_size` in notebook; length of `popt` |
| **Preference type k** | Integer 0, 1, 2 in `popt` |
| **Fixed type distribution** | `percent_agents_per_type`; α sweep via `payoff_alphas`; β sweep via `cents_offset` |
| **Strategy profile string of length K+θ** | Row of `profiles_indexed`; encoded as integer via `repmultipliers` |
| **Signal broadcast in dim d** | `profiles_indexed[x][d]` for profile x; `sigurns[d][i]` for RL |
| **Action conditioned on signal s** | `profiles_indexed[x][sigdimensions + s]`; `recurns[i][s]` for RL |
| **N_{k,t}(x)** (profile count per type) | `profile_count_typed[k][x]` |
| **M(i)** (total agents with profile i) | `profile_count_untyped[i]` |
| **U_k(x)** (expected utility) | `profile_utility_bytype[k][x]` from `executilcompute` |
| **Avg(U_k)** (type-average utility) | `typesaverage[k]` in `genBSreplicator_single_tstep` |
| **Replicator equation** | `genBSreplicator_single_tstep` |
| **Random initialization** | `genBSreplicatorexec_k_step` (replicator); `inertia=1` in notebook (RL) |
| **Mutation probability m0** | `mutateratef[0]`; applied via `muterandoms10_smart` |
| **Mutation probability m1** | `mutateratef[1]`; applied per-position in `repmutation_smart` |
| **"Urns" (one per profile position)** | `sigurns[d][i]`, `recurns[i][s]` |
| **"One ball per action"** | `inertia = 1` initialization in RL notebook |
| **"Draw one ball"** | `random2picks` → `agentprofilepicks` |
| **"Single profile for entire timestep"** | `agentprofilepicks` held fixed during `executilcomputeREforget` |
| **"Reinforce urns that contributed"** | Selective urn updates in `executilcomputeREforget` (only signal dims and receiver urn that was used) |
| **"Payoff-many balls"** | `Areinforcement = Ahfmultiplier × (coord_pref + sigcost)` |
| **Forgetting factor φ** | `forgetf = 0.998`; applied at start of `executilcomputeREforget` |
| **"Floor at 1"** | `if urn < 1: urn = 1` after reinforcement in `executilcomputeREforget` |
| **Signal 0 = not attending** | `profiles_indexed[x][d] == 0`; null-masking in `executilcompute` / `executilcomputeREforget` |
| **"Interacts as if others broadcast 0"** | Null-masking: zero any dim where either agent's signal is 0 |
| **Signal cost c per dimension** | `sigcostf × count_nonzero_dims`; added to every payoff in utility/RL loop |
| **H(h, x, i)** (assortment multiplier) | `assort_multipliers[assortsigA][assortsigB]` from `find_assort_multipliers` |
| **S(h, x, j) = (2d)^h** | `base_connection_weights[d] ** homophily_factor` in `execassort_array_create` |
| **Σ_j S(h,x,j) normalization** | Division by `Σ_k assort_array[i][k] × assort_sig_counts[k]` in `find_assort_multipliers` |
| **Only non-null dims counted for S** | `execassort_array_create`: skips dims where either signal is 0 |
| **Coordination preferences p_{xi,k}** | `coordination_preferencesf[type][action]` — 2D array indexed by preference type and action |
| **Coordination failure payoff = 0** | `genBSpunishf = 0` |
| **Optimal outcome (>75% threshold)** | `simple_stat_002` in `genBSexec_results` / `genBS_resultsRE` |
| **Cross-run signal relabeling** | `getalphasignaling` / `getalphastyped0` |
| **Mutual information I(S;A)** | `mutual_info` / `average_mutual_info` |

---

## Part 7: Implementation Details Not Stated in the Paper

These are computationally necessary choices that the paper leaves implicit, along with where
they appear in the code.

| Detail | Code Location | Explanation |
|:---|:---|:---|
| **Cascade integer rounding** | `genBSreplicator_single_tstep` | Converts float profile counts to integers via cumulative `np.around`, preserving type totals exactly. Prevents population drift. |
| **Zero-count correction** | `flip7zero_correction` | When rounding drives a type's profile count to zero (≈1-in-10,000), reinitializes that type's profiles randomly. |
| **Forgetting before reinforcement** | `executilcomputeREforget` | φ is applied to all urns at the function's start, then payoffs are added. Not stated explicitly in Appendix B. |
| **All-pairs interaction in RL** | `executilcomputeREforget` loop | Every agent pair interacts each timestep (not random sampling), consistent with "large representative sample" assumption. |
| **Pre-batched random draws** | `muterandoms10_smart`, `picks10` | 10 rounds of random data pre-generated for efficiency; refreshed every 10 timesteps. Numba JIT performance optimization. |
| **ε offset in sampling** | `random2picks`, `social_draws` | Adds `float_info.epsilon` to uniform random draws, ensuring samples from (0, 1] not [0, 1). Prevents degenerate zero-index always being selected. |
| **`execurnsf` inferred at reporting time** | `execrepconvert`, `execrepconvertRE` | Not a dynamic tracking variable; attention flags are inferred from urn argmax for output only. |
| **Coordination preferences ÷ 200 in RL** | RL notebook | Scales payoffs to ≈0.005–0.0075, matching Appendix B Table 1. Keeps reinforcement increments small relative to initial urn weight of 1. |
| **Convergence check interval: 100 vs 1000** | Both main loops | Replicator: 100-step exact equality check on deterministic profile arrays. RL: 1000-step interval because stochastic draws from stable urns still differ each timestep. |
| **`agentprofiles` vs `agentprofilepicks`** | Replicator vs RL | `agentprofiles` is a persistent per-agent profile index (updated each replicator step). `agentprofilepicks` is a fresh draw each RL timestep — different lifetime, different shape, different model. |
| **Signal cost applied to failures too** | `executilcompute`, `executilcomputeREforget` | `totalAsigcost` is added to both coordination payoffs and failure payoffs (genBSpunishf + sigcost). Attending agents pay the cost on every interaction regardless of outcome. |
| **`sigcostf` is negative** | Both notebooks | e.g., −0.0005. Adding a negative number to payoffs reduces them, implementing the cost deduction. |
| **`f` suffix convention** | Both scripts | Variable names ending in `f` (e.g., `population_sizef`, `sigcostf`) are numba JIT function parameter names. Conceptually identical to notebook-level variable names without the suffix. |

---

## Part 8: Legacy and Inactive Functions

The following functions exist in one or both scripts but are **not called** by either active
notebook's primary simulation path. Treating any of these as active leads to incorrect
descriptions of the model.

### In the Replicator Script (`FNs_...FLIP7...SMARTmutation.py`)

| Function | Why It Exists | Why It Is Inactive |
|:---|:---|:---|
| `utilcompute` | Non-exec utility calculation | Lacks signal cost; used in older non-exec replicator variants |
| `replicator_initialize` | Non-exec initialization | No `execurnsf`; superseded by `replicatorexec_initialize` |
| `repconvert` / `reconvert` | Non-exec profile-to-urn conversion | Superseded by `execrepconvert` |
| `genBSreplicator_full_play` | Non-exec main replicator loop | Superseded by `genBSreplicatorexec_full_play` |
| `genBSreplicator_first_tstep` | Alternative initializer | **Commented out** in active code; replaced by `genBSreplicatorexec_k_step` |
| `connections_init` | Per-agent weight matrix for explicit pairing | Superseded by analytical `execassort_array_create` / `find_assort_multipliers` |
| `pairings` | Weighted random pair sampling | Superseded by all-pairs analytical computation |
| `social_draws` | Per-agent signal sampling from urns | Superseded by `random2picks` in RL; irrelevant in replicator (profiles are deterministic) |
| `receiver_draws` | Per-agent action sampling from urns | Same as above |
| `genBS_check_success` | Per-pair payoff check for RL urns | Legacy pair-based RL function |
| `greetings_check_success` | Simpler per-pair payoff check | Legacy simpler model |
| `greetings_single_tstep` | Legacy single-step RL | Not used |
| `genBS_single_tstep` | Legacy single-step RL | Not used |
| `greetings_full_play` | Legacy full-play RL | Not used |
| `genBS_full_play` | Legacy full-play RL | Not used |
| `greetings_results` | Legacy result reporting | Superseded by `genBSexec_results` |
| `genBS_results` | Non-exec result reporting | Superseded by `genBSexec_results` |
| `muterandoms10` | Single-rate mutation | Superseded by `muterandoms10_smart` (two-rate) |
| `repmutation` | Single-rate mutation application | Superseded by `repmutation_smart` |

### In the RL Script (`FNs_...BZforget3...RothErev...py`)

| Function | Why It Exists | Why It Is Inactive |
|:---|:---|:---|
| `executilcomputeRE` | RL update without forgetting | Superseded by `executilcomputeREforget` |
| `connections_init`, `pairings`, `social_draws`, `receiver_draws` | Pair-based RL machinery | Superseded by `random2picks` + all-pairs `executilcomputeREforget` |
| `genBS_check_success`, `greetings_check_success` | Per-pair payoff for pair-based RL | Not called by `genBS_f7_RothErevExec_full_play` |
| `greetings_single_tstep`, `genBS_single_tstep` | Legacy single-step RL | Not used |
| `greetings_full_play`, `genBS_full_play` | Legacy full-play RL | Not used |
| `greetings_results`, `genBS_results` | Legacy result reporting | Superseded by `genBS_resultsRE` |

### Why These Functions Exist

These legacy functions represent the evolutionary history of the codebase. Earlier model
versions implemented RL via explicit random pairing (one pair per interaction event per
timestep, drawing signals and actions from urns for each event). The final active models
replaced this with:
1. **For the replicator**: fully analytical utility computation over all profile pairs,
   avoiding any per-agent stochastic elements within a timestep.
2. **For the RL model**: a single strategy draw per agent per timestep (`random2picks`),
   followed by analytical all-pairs interaction (`executilcomputeREforget`). This is the
   computational tractability improvement explicitly noted in Appendix B.

The paper itself provides the justification for why the pair-based functions are inactive:
"We found that having agents select a single strategy profile that they employ for all
interactions on a single timestep made this learning dynamic significantly more computationally
tractable. We also suspect that it improves agents' ability to learn in virtue of reducing the
amount of noise in the system."
