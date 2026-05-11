# Code–Paper Correspondence: Social Identity Signaling Simulations

This document maps every subfunction in both simulation codebases to its corresponding description in the paper ("The Evolution of Identity Signals for Coordination in Diverse Societies") and its supplement (Appendix A–B), and supplies written descriptions for implementation details not covered explicitly in those texts.

---

## Part 1: Which Notebook Is Which Learning Dynamic

**`genBS_v0055k_assort_FLIP7_repEXEC_sm_sweep_Merced_topA-Copy4.ipynb`** implements the **discrete replicator dynamics** described in Section 2.2 of the main paper. It imports `genBSreplicatorexec_full_play_typed` from `FNs_genBachStravinsky_v0055k_assort_FLIP7_rep_execNULLsig_SMARTmutation.py`. The parameter settings in the notebook (1,000 agents, signal cost −0.0005, mutation rates m0=0.01, m1=0.1) match the single-embedding parameters in Table 4 / Section 4.2 of the main paper. The coordination preferences in the sweep match Table 7.

**`genBS_v0055k05_BZforget3_assort_f7_RothErevExec_sweep_Merced_topA-Copy34.ipynb`** implements the **Roth-Erev reinforcement learning with forgetting** described in Appendix B of the supplement. It imports `genBS_f7_RothErevExec_full_play_typed` from `FNs_genBachStravinsky_v0055k05_BZforget3_assort_f7RothErev_rep_execNULLsig.py`. The parameter settings (500 agents, forget factor 0.998, signal cost −0.000125, coordination preferences scaled by 1/200) match Appendix B's Table 1 and surrounding discussion.

The two `.py` files share a large body of utility functions (analysis, assortment, population setup, output formatting). The key difference is in how each timestep updates agent strategies: the replicator file uses `executilcompute` → `genBSreplicator_single_tstep`, while the RL file replaces that pair with `random2picks` → `executilcomputeREforget`.

---

## Part 2: Shared Infrastructure

The following functions appear in both `.py` files and implement model components common to both learning dynamics.

### `population_array(population_size, percent_agents_per_type)`

**Paper connection (§2.2):** "We initialized a population of N = 1,000 agents with a fixed distribution of preference types."

Creates a flat integer array `poptypes` of length `population_size` where each entry is the type index (0, 1, 2, …) of the corresponding agent. Agents of the same type are stored contiguously. The input `percent_agents_per_type` (e.g., `[0.33+β, 0.33−β, 0.34]`) determines the fraction of each type. This function is not called inside any numba JIT loop — it runs once before the simulation begins to fix the population composition.

### Strategy profile representation: `profiles_indexed`, `repmultipliers`, `dimactions`

**Paper connection (§3.1, §4.1):** "Agents' strategy profiles can be expressed as strings of length 1 + θ … The first element in this string represents the social signal that an agent broadcasts and the subsequent elements represent the action that is played when paired with an agent who broadcasts the corresponding signal."

The paper's string representation is implemented as follows. Each strategy profile is a vector of length `sigdimensions + numsignals` (here 2 + 4 = 6 for the two-dimension, two-signal-per-dimension case in the single-embedding sweep). The first `sigdimensions` entries hold the signal broadcast in each dimension (0 = null/no signal, 1 or 2 = an active signal). The next `numsignals` entries hold the action taken when observing each of the possible combined signal indices (0–3 in the 2×2 case). This directly implements the paper's description: "the first element … the social signal … the subsequent elements … the action that is played when paired with an agent who broadcasts the corresponding signal."

`dimactions` lists the number of possible values at each position in the string: `[signals_in_dim_1, signals_in_dim_2, actions, actions, actions, actions]`. `repmultipliers` is the mixed-radix positional weight for each position, computed as a shifted cumulative product of `dimactions`. This maps each integer profile index uniquely to a profile string and back, enabling efficient storage of strategy profiles as single integers rather than full vectors.

`profiles_indexed` is a lookup table of shape `(numprofiles, sigdimensions + numsignals)`, constructed inside the initialize functions, enumerating all possible strategy profiles in order. Entry `profiles_indexed[k]` gives the full string for profile k.

### `replicator_initialize` and `replicatorexec_initialize`

**Paper connection (§2.2):** "We always begin simulations with agents whose strategy profiles are randomly assigned."

**`replicator_initialize`** (used for the non-exec replicator) and **`replicatorexec_initialize`** (used for the exec replicator and also within the RL code to set up `profiles_indexed` and `sigperdim`) both: (a) set each agent's signal urn to a one-hot vector drawn uniformly at random from all signal options per dimension, (b) set each agent's receiver urn to a one-hot vector drawn uniformly at random from all actions, and (c) enumerate all possible strategy profiles into `profiles_indexed`. The one-hot representation of random initialization maps to the paper's "randomly assigned strategy profiles" — each agent starts with a single definite strategy rather than a distribution over strategies.

**`replicatorexec_initialize`** additionally allocates an `execurns` array (shape `[population_size, sigdimensions]`) that records whether each agent attends to each signaling dimension. This is inferred from the signal urn (null signal = not attending) and is used mainly for result reporting, not during the simulation dynamics themselves.

### Signal and action urn representation

**Paper connection (§3.3, Appendix B):** "Each urn begins a simulation with one ball corresponding to each possible action for that location in the strategy profile string."

Throughout both codebases, agent state is stored in two main arrays:

- **`sigurns`**: a Python list of `sigdimensions` numpy arrays, each of shape `[population_size, signals_per_dim]`. Entry `sigurns[d][i][s]` is the weight (ball count) for agent `i` broadcasting signal `s` in dimension `d`. In the replicator, these are always one-hot. In the RL model, they begin at 1 for all entries and evolve continuously.

- **`recurns`**: a numpy array of shape `[population_size, numsignals, numactions]`. Entry `recurns[i][signal_combo][action]` is the weight for agent `i` playing `action` upon observing combined signal index `signal_combo`. Again one-hot in the replicator, continuous in RL.

### Signal cost

**Paper connection (§3.3):** "a signal cost, c, that an agent incurs if she broadcasts any signal other than 0. When an agent uses a nonzero signal, its payoff from an interaction is decreased by c."

In the notebook, `signal_cost` is set to a negative number (e.g., −0.0005). In `executilcompute` and `executilcomputeREforget`, the cost is applied by computing `totalAsigcost = (number of non-null signal dimensions) × sigcostf`. This is then *added* to the raw payoff — and because `sigcostf` is negative, the net effect is a deduction. Crucially, signal cost is applied on every interaction regardless of whether it results in successful coordination, matching the paper's statement that payoff is decreased for any agent using a nonzero signal.

One detail not stated explicitly in the paper: the cost is proportional to the number of active signaling dimensions, not simply a flat cost for "using signals at all." This means agents attending to both signaling dimensions pay cost 2c per interaction. This is consistent with the paper's statement (§4.1) that "we assume the signal cost c is incurred for each dimension attended to."

### Null-signal interaction rule

**Paper connection (§3.3):** "When an agent does not attend to signals (i.e., when she broadcasts 0), she interacts with all other agents as if they had also broadcast 0."

In `executilcompute` and `executilcomputeREforget`, before computing which action A takes in response to B's signal, the code zeros out any dimension where *either* agent has a null signal (0) in that dimension:
```python
for idx1212 in range(0, sigdimensionsf):
    if Aprofile_copy[idx1212] == 0 or Bprofile[idx1212] == 0:
        Aprofile_copy[idx1212] = 0
        Bprofile[idx1212] = 0
```
The effective "signal seen by A" is then computed from B's zeroed-out profile. If B broadcasts null in all dimensions, the signal index seen by A is 0 in all dimensions, so A looks up `recurns[A][0]` — their action for signal combination 0, which is exactly "treating them as if they broadcast 0." The symmetry (zeroing both A and B if either is null) ensures mutual blindness.

### Assortment / homophily

**Paper connection (Appendix A, Appendix B):** H(h,x,i) = N / Σ_j [S(h,x,j)] × S(h,x,i), where S(h,x,j) = 2^h for 1D signals (same signal), and for multiple dimensions S(h,x,j) = (2×d)^h if the same signal in d dimensions, 1 otherwise.

The assortment computation is split across three functions:

**`assort_array_create` / `execassort_array_create`**: Builds a static `[numsignals × numsignals]` lookup table of raw assortment weights. For signal pair (i, j), counts the number of signal dimensions where their values agree, then computes `base_connection_weights[d]^homophily_factor`. The `base_connection_weights` array is `[1, 2, 4, 8, 16, 32]`, so entry `d` gives 2^d. Together with the homophily exponent, this gives `(2^d)^h = 2^(d×h)`. For d=1 and h=1, this yields 2, matching S(h,x,j) = 2^1 = 2 for one matching dimension. For d=0, `2^0 = 1` matching the baseline.

The exec variant (`execassort_array_create`) differs in one respect: it only counts dimensions where *both* signals are nonzero when tallying shared dimensions. This captures the intuition that two agents both broadcasting null in a dimension do not share a meaningful signal there.

**`get_sig_counts` / `get_sig_countsRE`**: Counts how many agents in the current population are broadcasting each combined signal index. This provides the Σ_j[S(h,x,j)] denominator needed for the normalization. The replicator version works from the profile count arrays; the RL version works from `agentprofilepicks` (the per-agent profile draws for the current timestep).

**`find_assort_multipliers`**: For each signal pair (i, j), computes the final assort multiplier as `assort_array[i][j] × (N / Σ_k assort_array[i][k] × count_k)`. This exactly implements N / Σ_j[S(h,x,j)] × S(h,x,i) from the paper. The resulting matrix `assort_multipliers[i][j]` is then multiplied into every utility/payoff computation involving agents with signal i interacting with agents with signal j.

### Convergence / early termination

**Paper connection (Table 4, footnote 2 of Appendix E):** "every 100 timesteps it was checked whether the distribution of agents' strategy profiles was unchanged. If so, the simulation was halted prior to reaching 4 × 10^4 timesteps."

Both full-play functions include this check, but with different periodicity. The replicator checks every 100 timesteps (`if idxN0%100 == 0`) by comparing `agentprofilesCOPY` to `agentprofiles`. The RL model checks every 1,000 timesteps, comparing `agentprofilepicksCOPY` to `agentprofilepicks`. The RL model uses a longer interval because its stochastic per-timestep profile sampling means that even a stable urn configuration will produce different `agentprofilepicks` from timestep to timestep, making exact equality less diagnostic.

### Output and results reporting

**`genBSexec_results` / `genBS_resultsRE`**: Called at the end of each simulation run (and periodically as snapshots). Computes:
- `socialsig_aggregate[type][signal]`: count of agents of each type using each combined signal index as their modal signal.
- `action_aggregate[type][signal][action]`: distribution of actions taken by agents of each type in response to each signal.
- `simple_stat_001[y_type][x_type][action]`: records the most likely action that agents of type y take when encountering the most common signal of type x (using argmax throughout).
- `simple_stat_002[y_type][x_type][action]`: a stricter version requiring that >75% of type x uses that signal AND >75% of type y's responses to that signal are that action. This is the threshold used to classify a simulation outcome as having a "clean" behavioral pattern.

The paper's classification of outcomes (iv), (v), etc. in Appendix A, and outcomes (i)–(xiii) in Appendix G, are computed in the notebooks by inspecting `simple_stat_002` arrays across runs.

**`time_update`**: Records time-series snapshots of the population's signaling and action distributions. This is used only for visualizing the evolution of individual simulation runs (Cells 12–13 in the replicator notebook). It is not used for any cross-run aggregate statistics reported in the paper.

**`getalphasignaling` / `getalphastyped0`**: Post-processing function called after all runs complete. Because the labeling of which signal is "signal 1" vs. "signal 2" is arbitrary across runs (the model has no fixed assignment of signals to groups), this function re-labels signals across runs so that type 0's (Lagashites') most common signal in a given dimension is consistently called "alpha." This enables meaningful aggregation of signal distributions across many runs. The function also records `execsruns`, indicating for each type which signaling dimensions they most commonly attend to (inferred from whether their modal signal in that dimension is null or not). This is not mentioned explicitly in the paper but is an analysis tool for understanding which dimensions each type uses.

**`mutual_info` / `average_mutual_info`**: Computes the Shannon mutual information between the signal broadcast by an agent and the action taken in response, averaged across the population. This is used to compute the synergistic information measure discussed in §4.3 of the paper in the context of intersectional identities. The paper cites Varley and Kaminski (2022) and partial information decomposition.

---

## Part 3: Replicator Dynamics — `genBSreplicatorexec_full_play` and Supporting Functions

This section covers all functions used in the replicator dynamics notebook and its `.py` file that are specific to the replicator path (i.e., not already covered in Part 2).

### Main loop: `genBSreplicatorexec_full_play`

**Paper connection (§2.2):** "The model dynamics proceeded in discrete timesteps, using replicator dynamics under the assumption of a large number of interactions in a well-mixed population."

This function orchestrates the entire replicator simulation. The main loop runs from timestep 1 to `runlength` (nominally 4×10^4) with an early-termination check every 100 steps. The calling convention `genBSreplicatorexec_full_play_typed` is a thin wrapper that converts Python lists to numba-typed lists and back, required for numba's JIT compilation.

### Initialization for replicator: `genBSreplicatorexec_k_step`

**Paper connection (§2.2):** "We always begin simulations with agents whose strategy profiles are randomly assigned."

After the urn-based initialization (which sets up `profiles_indexed`), `genBSreplicatorexec_k_step` assigns each agent a profile index uniformly at random from all `numprofiles` possible profiles. This is conceptually equivalent to the paper's random initialization — every possible strategy is equally likely to be assigned. The result is stored in `profile_count_typed[type][profile]` and `profile_count_untyped[profile]` (counts of agents of each type with each profile) plus `agentprofiles[agent]` (the profile index for each individual agent).

### Utility computation: `executilcompute`

**Paper connection (§2.2):** "The utility of strategy profile x for preference type k is calculated as: Uk(x) = Σ_{i∈Y} [M(i) × p_{xi,k}] if x ≠ i, Σ_{i∈Y} [(M(i)-1) × p_{xi,k}] if x = i … where M(i) is the number of agents (of any preference type) in the population who play strategy i, and p_{xi,k} is the payoff to an agent of preference type k for playing strategy x when paired with an agent who plays strategy i."

**Appendix A:** "U(x) = Σ_{i∈Y} [H(h,x,i) × M(i) × p_{xi}] if x ≠ i, Σ_{i∈Y} [H(h,x,i) × (M(i)-1) × p_{xi}] if x = i"

`executilcompute` implements the assortment-weighted utility function. For every pair of distinct profile types (A, B):
1. Computes the assortment multiplier H(h,A,B) = `assort_multipliers[assortsigA][assortsigB]` using the signal indices before null-zeroing (assortment depends on what signals agents actually broadcast, not on the effective signal seen after null-matching).
2. Applies null-signal zeroing to determine effective signal exchange.
3. Looks up A's action (from B's effective signal) and B's action (from A's effective signal).
4. Adds `hfmultiplier × M(B) × (coordination_preference[type][action] + signal_cost)` to the utility of profile A for each type (using M(B)-1 when A = B to exclude self-pairing). The signal cost is negative, effectively reducing payoffs.
5. When actions differ (coordination failure), `genBSpunishf` replaces the coordination preference. In the single-embedding sweep, `genBSpunish = 0`, so failures yield zero payoff.

The result is `profile_utility_bytype[type][profile]`, the total utility that an agent with profile A would accumulate across all its interactions with the current population, computed separately for each preference type.

### Replicator update: `genBSreplicator_single_tstep`

**Paper connection (§2.2):** "N_{k,t+1}(x) = N_{k,t}(x) + N_{k,t}(x) × [Uk(x) - Avg(Uk(i))_{i∈X}]"

This function applies the discrete replicator equation. For each preference type k:
1. Zeros out utilities for profiles not currently present in type k's population (utilities are only meaningful for present profiles).
2. Computes the average utility across all present profiles of type k.
3. Updates each profile count: `profile_count_typed[k][x] += profile_count_typed[k][x] × (utility[k][x] - type_average[k])`.
4. Clamps any negative counts to zero (an artifact of discrete rounding when a profile's utility is far below average).
5. Renormalizes so the total count of each type remains constant at `typestotal[k]`.
6. Converts floating-point counts back to integer agent assignments through cumulative rounding, building `agentprofiles` as an ordered array where agents of each profile type are listed contiguously.

The renormalization (step 5) and the integer rounding (step 6) are implementation details not stated in the paper's equation but necessary for a discrete-population simulation.

### Zero correction: `flip7zero_correction`

**Paper connection:** Not mentioned explicitly, but this corrects a rare numerical artifact.

When the replicator dynamics collapse all agents of a given type into profiles with utilities so far below average that rounding produces zero agents for that type, this function rescues the simulation by randomly reinitializing that type's profile distribution. The code notes this is "a 1 in 10,000 chance" event. Without this correction, certain runs would crash or produce invalid results.

### Mutation: `muterandoms10_smart` and `repmutation_smart`

**Paper connection (§2.2):** "We also allow for the possibility that an agent's strategy profile is not copied faithfully, or 'mutates.' This is governed by two parameters, m0 and m1. Each agent is selected for mutation with probability m0. If selected, each element in the string expression of the agent's strategy profile is assigned a random value from among the allowable values with probability m1."

**`muterandoms10_smart`** pre-generates 10 timesteps' worth of mutation draws at once (for efficiency):
- `mutations10[t][agent]`: a Bernoulli draw with success probability m0, marking whether agent is selected for mutation at timestep t.
- `newprofiles10[t][k]`: pre-drawn target profile indices for the k-th mutation event at timestep t.
- `muteprofilestrings10[t][k][position]`: per-position Bernoulli draws with success probability m1, indicating which positions in the profile string actually get mutated.

**`repmutation_smart`** applies these pre-drawn mutations. For each agent selected for mutation, it constructs a new profile by mixing the old profile with the randomly drawn target profile: each position uses the random profile's value with probability m1 (where `muteprofilestrings10 == 1`), or keeps the old profile's value otherwise. This is the "smart" mutation: instead of replacing the entire profile, only m1-fraction of positions are mutated, giving continuity between parent and child profiles. The result is then converted to a profile index using `repmultipliers` and the relevant profile count arrays are updated. Note that `agentprofiles` is NOT updated during mutation — it is rebuilt from scratch each timestep by `genBSreplicator_single_tstep`.

The pre-generation batch of 10 and the refresh every 10 timesteps (`if idxN0%10 == 0`) is a performance optimization that avoids numba's overhead from calling the random number generator on every timestep.

### Format conversion: `repconvert` and `execrepconvert`

Not mentioned in the paper; these are implementation utilities.

`repconvert` and `execrepconvert` translate from the replicator's compact profile-index representation back to the urn-based representation (`sigurns`, `recurns`, `execurns`). This conversion is needed when the simulation wants to call reporting functions that expect urn format (such as `genBS_results` or `time_update`). The exec version additionally fills in `execurns` by reading whether each agent's signal in each dimension is null (0 = not attending) or nonzero (1 = attending).

---

## Part 4: Roth-Erev Reinforcement Learning — `genBS_f7_RothErevExec_full_play` and Supporting Functions

This section covers all functions specific to the RL path.

### Main loop: `genBS_f7_RothErevExec_full_play`

**Paper connection (Appendix B):** "nothing about the Generalized Bach or Stravinsky game requires agents to learn according to replicator dynamics. In this Appendix, we give some limited simulation results for the model in which the discrete replicator dynamics is replaced by agents learning according to Roth-Erev reinforcement learning with forgetting."

This function is the RL counterpart to `genBSreplicatorexec_full_play`. Its overall structure is parallel — it runs for up to `runlength` timesteps, checks convergence, records snapshots — but the per-timestep update mechanism is entirely different. The key loop comment in the code makes the replacement explicit: "replacing replicator with Roth-Erev."

**Important**: the RL main loop does NOT call any mutation function. The parameter `mutateratef` is passed to the function but unused within `genBS_f7_RothErevExec_full_play`. Exploration in the RL model comes from two other sources: (1) the stochastic sampling from urns each timestep via `random2picks`, and (2) the floor constraint that keeps every urn option at weight ≥ 1, ensuring no action is ever permanently eliminated. This replaces explicit mutation.

### Initialization: `RothErevExec_initialize`

**Paper connection (Appendix B):** "each agent in the population having an urn for each location in the string representation of their strategy profile. Each urn begins a simulation with one ball corresponding to each possible action for that location in the strategy profile string."

Unlike the replicator's `replicatorexec_initialize`, this function does NOT set urns to one-hot vectors. Instead, the notebook sets all urn entries to 1 via `inertia = 1` before calling the simulation. `RothErevExec_initialize` computes only the structural variables `sigperdimf` and `profilecaps` (the size of each urn), without touching the urn contents. The urn entries of 1 at all positions correspond to the paper's "one ball corresponding to each possible action" — uniform initial probability over all choices.

### Per-timestep profile drawing: `random2picks`

**Paper connection (Appendix B):** "In a single timestep, an agent begins by drawing one ball from each of their urns at random with equal probability. These draws then determine the strategy profile that the agent employs for the entire timestep while interacting with every other agent in the population."

`random2picks` performs the stochastic profile draw for all agents simultaneously. For each agent i and each signal dimension d, it draws one signal value from the signal urn proportionally to the urn weights (using the cumulative-sum / threshold method). Similarly, for each possible incoming signal combination (0 through numsignals-1), it draws one action from the corresponding receiver urn. The result is stored in `agentprofilepicks[agent][position]`, a profile-like array holding the specific signal and action choices made by each agent for this timestep.

This implements the paper's "drawing one ball from each urn" — the probability of drawing option s from an urn is proportional to the weight of option s in that urn. The pre-generated `picks10` array (10 batches of random draws, regenerated every 10 timesteps) provides the randomness. Using pre-generated randoms is more efficient than calling the RNG inside the tight loop.

### RL update with forgetting: `executilcomputeREforget`

**Paper connection (Appendix B):** "Reinforcements can be thought of as adding additional balls of the type that was drawn to the urn that it was drawn from … at the end of a timestep, forgetting is implemented by multiplying the number of balls of each type by a forgetting factor 0 < φ < 1. For any ball type whose quantity falls below 1 in any urn, that quantity is raised to 1."

**Appendix B on assortment:** "To represent assortment, we adjusted agents payoff by multiplying it by H(h,x,i)."

This function is the heart of the RL model. It replaces both `executilcompute` and `genBSreplicator_single_tstep` from the replicator.

**Step 1 — Forgetting**: Before any reinforcement, every urn entry is multiplied by `forgetf` (φ = 0.998). This is applied to both signal urns and receiver urns.

**Step 2 — Reinforcement**: For each ordered pair (A, B) where A < B (the loop structure `for idx1201mod in range(1, population_size - idx1200)` with modular indexing ensures each unordered pair is processed once, with both agents reinforced in the same call), the function:
- Reads A's and B's profiles from `agentprofilepicks` (the draws made at the start of this timestep).
- Determines effective signals after null-zeroing.
- Determines the actions A and B take.
- Computes the assortment multiplier `Ahfmultiplier` for A's signal pairing with B.
- If A and B coordinate (same action): `Areinforcement = Ahfmultiplier × (coordination_preference[Atype][action] + signal_cost)`.
- If not: `Areinforcement = Ahfmultiplier × (genBSpunish + signal_cost)`. With genBSpunish=0, failed interactions contribute zero reinforcement before the signal cost, meaning signal cost alone is still applied.
- Adds `Areinforcement` to A's signal urn entries for each dimension (for the signal A actually broadcast this timestep) and to A's receiver urn entry (for the action A took given B's signal).

**Step 3 — Floor enforcement**: After all interactions, any urn entry below 1 is raised to 1.

A key implementation note not stated in the paper: because this function interacts A with every other agent in the population (not random pairs), the RL model implicitly assumes "a large representative sample" of interactions per timestep, consistent with the paper's general modeling assumption but achieved differently than pairwise sampling. The paper's Appendix B states: "We found that having agents select a single strategy profile that they employ for all interactions on a single timestep made this learning dynamic significantly more computationally tractable." The `agentprofilepicks` array is what makes this tractable — all interactions use the same pre-drawn profile, so we only need to loop over agent pairs and add reinforcements, rather than re-sampling from urns for each individual interaction.

The coordination preference values in the RL notebook are scaled by 1/200 compared to the replicator notebook. This keeps the reinforcement increments small relative to initial urn weights (1 ball each), preventing early interactions from irreversibly dominating urn distributions before agents have explored the space.

### RL-specific result reporting: `genBS_resultsRE` and `execrepconvertRE`

Not explicitly described in the paper; these are implementation utilities for extracting results from the RL model.

**`execrepconvertRE`**: Converts the continuous urn representation to the modal strategy profile used for reporting. For each agent, it reads the argmax of each signal urn (the signal with the highest weight) and maps this to a 0/1 exec value (0 if the modal signal is null, 1 if not). This produces a point estimate of the agent's current "effective strategy" for output purposes only, without affecting the simulation dynamics.

**`genBS_resultsRE`**: Analogous to `genBSexec_results` from the replicator, but designed to work directly from the urn-based representation that the RL model maintains. It takes `sigurnsf_4report` and `recurnsf_4report` produced by `execrepconvertRE`. The internal logic of aggregating `socialsig_aggregate`, `action_aggregate`, `simple_stat_001`, and `simple_stat_002` is the same as in the replicator's reporting function.

### Remaining RL-only function: `get_sig_countsRE`

Not mentioned in the paper; this is an implementation utility.

`get_sig_countsRE` tallies the number of agents using each combined signal index in the current timestep by iterating over `agentprofilepicks` rather than `profile_count_untyped` (which the replicator's `get_sig_counts` uses). This is necessary because the RL model does not maintain running counts of how many agents hold each profile — instead, profiles are drawn fresh each timestep. The result feeds directly into `find_assort_multipliers`.

---

## Part 5: Functions Present in Both Files but Only Used in the RL Full-Play Path in the BZforget File

Both `.py` files contain `genBSreplicator_full_play` (replicator without exec, i.e., without the null-signal attention mechanism), `genBS_full_play` (a lightweight RL-style loop without forgetting), `greetings_full_play` (an even simpler model without generalized payoffs), and `genBSreplicatorexec_full_play` (replicator with exec). These functions represent development history and earlier model versions. In the BZforget notebook, only `genBS_f7_RothErevExec_full_play_typed` is actually called; the others are present but unused in the active sweep cell. Similarly, the FLIP7 notebook only calls `genBSreplicatorexec_full_play_typed`.

The function `executilcomputeRE` (without forgetting) is present in the BZforget file but not called by the active RL function — `executilcomputeREforget` is used instead. The forgetting version simply applies the φ multiplier to all urns before reinforcement; the non-forgetting version applies reinforcement directly.

---

## Part 6: Model Description Components Without Direct Code Connections, and Code Components Without Direct Paper Connections

### Paper descriptions that connect only implicitly

**"Agents do not use signals assortatively to find interaction partners" (§3.2):** This is the non-assortment baseline. It corresponds to setting `homophily_factor = 0` in the notebooks. With h=0, all entries in `assort_array` equal `base_connection_weights[d]^0 = 1` regardless of d, so `assort_multipliers` becomes a flat matrix of 1/numsignals for all pairs, equivalent to random mixing. The FLIP7 notebook uses `homophily_factor = 1`, the BZforget notebook uses `homophily_factor = 0` in its main sweep.

**"T levels of nested embeddings in the group structure require T+1 dimensions of social signals" (§4.2):** This is a theoretical claim derived from the simulation results, not a code parameter. The number of signaling dimensions is set by `sigdimensions = 2` in both notebooks.

**"strategy profiles are transmitted" (§2.2, §3.1):** In the replicator, strategy transmission is implemented implicitly through the replicator equation — successful strategies become more common without any explicit copying mechanism. There is no function that "copies" a strategy from one agent to another.

### Code details not explicitly in the paper

**`sigdimensions + numsignals` (not `K + θ`) profile length:** The paper's §4.1 states profiles are strings of length K + θ where K is the number of signal dimensions and θ is the number of possible signal combinations. The code computes θ = `np.prod(numsignals_perdim)` (e.g., 2×2 = 4) and stores profiles as arrays of length `sigdimensions + numsignals` = K + θ. This exactly matches the paper.

**`base_connection_weights` array:** This is `[1, 2, 4, 8, 16, 32]` in both notebooks. The index d gives 2^d, which together with `homophily_factor` implements S(h,x,j) = (2^d)^h. This array is technically a generalization allowing non-power-of-2 weights, but in practice the notebooks always use powers of 2 matching the paper's formulation.

**Profile count normalization order:** After applying the replicator equation, the code normalizes `profile_count_typed` to maintain constant type sizes and then sums to get `profile_count_untyped`. This order (normalize first, then sum) avoids rounding inconsistencies between the typed and untyped counts.

**Convergence criterion details:** The paper says "whether the distribution of agents' strategy profiles was unchanged." In code, this is `np.array_equal(agentprofilesCOPY, agentprofiles)` — an exact match of the integer profile index array for every agent, not just aggregate distributions. This is a very strict criterion (distribution-exact, not just statistically stable), appropriate for a system that is expected to reach a true absorbing state under the replicator dynamics.

**Snapshot recording:** Both full-play functions record intermediate `socialsig_aggregate` snapshots at regular intervals during the simulation (`signal_snapshots` parameter, set to 1 in both sweep notebooks). With `signal_snapshots=1`, only one snapshot is taken at the very end (in addition to the final state), and the sweep analyses are all based on final state. The snapshot infrastructure supports more detailed time-series analysis in exploratory cells of the notebooks.
