**Prompt:** Please look through Sonnet's "difinitive mapping" and try to identify any oversights related to things done in the code that are not documented in the paper.



**Opus 4.7 response:**

That's a fair request — looking back at my definitive mapping with fresh eyes, there are several things the code does that the paper doesn't document which I either missed entirely or only mentioned superficially. Let me work through them honestly.

## Major Oversights — Things I Should Have Flagged More Prominently

### 1. Signal cost is incorporated INTO reinforcement, not applied separately

In `executilcomputeREforget`, the reinforcement value is computed as:

```
Areinforcement = Ahfmultiplier × (coordination_payoff + totalAsigcost)
```

With `genBSpunish = 0` and a negative `sigcostf`, this means **failure interactions for signaling agents produce negative reinforcement**, not just absence of reinforcement. The paper says "if an agent receives a non-zero payoff the urns that contributed are reinforced" — but the code's formulation reinforces with the *net* of payoff and cost, which is negative for failures with signaling. The floor=1 rule then partially absorbs this. I noted that the cost is applied to failures, but I didn't articulate the asymmetry this creates: positive reinforcement accumulates freely, negative reinforcement bottoms out at 1.

### 2. The `mutual_info` function computes Shannon MI, not synergistic information

This is a potentially substantive code-paper discrepancy I missed. Section 4.3 of the paper discusses results using **partial information decomposition** and **synergistic information** following Williams and Beer (2010) and Varley and Kaminski (2022). The code's `mutual_info` / `average_mutual_info` functions compute standard Shannon mutual information I(S; A). These are different quantities. Either:

- There is separate analysis code outside these scripts that computes PID, or
- The synergistic information was computed externally from the simulation outputs, or
- Shannon MI is being used as a proxy.

I should have flagged this rather than uncritically equating the two.

### 3. `getalphasignaling` performs a non-trivial cross-run alignment

I mentioned this function relabels signals for cross-run consistency, but I undersold the consequence: **without this step, all cross-run heatmaps in the paper would be uninterpretable**, because in any individual run, "signal 1" and "signal 2" in a given dimension are arbitrarily labeled. The function identifies type 0's modal signal per dimension per run and consistently relabels everything else relative to it. This is a substantial analysis step the paper does not describe, and I should have emphasized it as such rather than treating it as a routine post-processing function.

### 4. The convergence criterion is much stricter than the paper implies

The paper says the simulation halts if "the distribution of agents' strategy profiles was unchanged." The code requires `np.array_equal(agentprofilesCOPY, agentprofiles)` — **exact per-agent integer equality** for every one of N agents. The distribution being unchanged is a weaker condition than every agent holding the same profile. In practice, this likely means early termination almost never fires in the replicator (any single agent's profile changing through mutation or replicator update breaks the equality), and essentially never fires in RL (where stochastic urn draws produce different `agentprofilepicks` each step). Most simulations probably run to the full 4×10⁴ timesteps.

### 5. The replicator's non-negativity clamp is an undocumented model decision

The replicator equation `N_{t+1}(x) = N_t(x) + N_t(x) × (U_k(x) − Avg(U_k))` can produce negative values when `U_k(x) << Avg(U_k)` (specifically when `U_k(x) − Avg(U_k) < −1`). The code clamps any negative `profile_count_typed[k][x]` to zero before renormalization. The paper doesn't address what happens in this case. The clamp is a substantive modeling choice — it means strategies that perform very badly are eliminated rather than going negative. This affects which trajectories the model can realize.

### 6. `flip7zero_correction` loses all evolutionary history

I mentioned this function exists, but I undersold what happens when it fires. When cascade rounding drives a type's total agent count to zero, it doesn't just nudge the system — it **uniformly randomly reinitializes that type's entire profile distribution**, throwing out whatever the population had learned for that type. This is rare (~1 in 10,000 by code comments) but when it does fire, it's a substantial intervention into the trajectory not anywhere in the paper.

---

## Substantive Oversights — Smaller but Still Real

### 7. `execurnsf` in the replicator becomes stale during the simulation

The replicator works at the level of integer profile indices, not urns. `replicatorexec_initialize` sets `execurnsf` based on the initial random assignment, but it is **not updated** during the simulation — `executilcompute` and `genBSreplicator_single_tstep` don't touch it. It is rebuilt fresh each reporting interval by `execrepconvert`. So `execurnsf` reflects the initial state until the first reporting interval, then jumps to reflect the current state. Mid-simulation, it could be wildly out of date. This doesn't affect simulation outcomes (since nothing reads it) but means the variable's value has no consistent meaning between reporting points.

### 8. `np.argmax` tie-breaking biases attention inference toward "not attending"

When `execrepconvertRE` (or `execrepconvert`) infers attention from `argmax(sigurns[d][i])`, ties are broken by selecting the lowest-indexed option. Since the null signal is at index 0, any agent whose urn weights are tied between null and a non-null signal is recorded as **not attending**. This produces a slight systematic bias in the reported attention statistics, especially early in RL simulations where urns are all near the inertia value.

### 9. The ordering within `genBSreplicator_single_tstep` is a specific choice

The function performs: equation update → non-negativity clamp → renormalize to preserve type totals → cascade integer rounding → rebuild `agentprofiles`. The paper specifies the equation and the renormalization but doesn't fix the order between them or the discretization step. A different ordering (e.g., round first, then normalize) would produce different trajectories.

### 10. The all-pairs RL loop processes each unordered pair exactly once and reinforces both agents

`executilcomputeREforget` uses a loop structure where each unordered pair (A, B) is visited once, and within that single visit, both A's urns and B's urns are reinforced from the single payoff computation. This is an optimization, but it has a subtle consequence: the reinforcement order matters within a timestep (A is processed before B in the pair), and the all-pairs loop visits pairs in a deterministic order. So agents whose indices come up later see slightly more recent versions of their partners' urns being reinforced first — though since urns are only mutated after the loop completes (forgetting happens before, floor happens after), this shouldn't actually create asymmetry. But the deterministic visit order is an implementation choice.

### 11. The `coordination_preferences / 200` scaling in the RL notebook

I mentioned this exists. What I didn't articulate is *why it's necessary*: with initial urn weights of 1, an unscaled coordination payoff of 1+α would multiply urn weights by ≈2 in a single interaction with assortment, completely dominating the prior. The /200 scaling brings the per-interaction urn delta to ~0.005, comparable in magnitude to the inertia, so learning happens gradually over many interactions rather than locking in on the first few. This is a calibration choice the paper presents as predetermined coordination preferences.

### 12. Type proportions use 0.34/0.33/0.33 (not 1/3 each) for clean integer counts

The notebooks set the Akkadian proportion to 0.34 and the other two to 0.33 (when β = 0). With `population_size = 1000` or 500, this gives clean integer counts (340/330/330 or 170/165/165). A 1/3 split would produce non-integer counts requiring rounding that would distort the intended proportions. Minor but real implementation detail.

### 13. Mutation timing within the replicator loop is undocumented

The paper says mutations happen but doesn't specify when within the timestep. The code applies mutation **after** the replicator update (and only every 10 timesteps). This means each replicator update reflects the post-mutation distribution from 10 steps ago plus 10 steps of replicator dynamics. A different ordering (mutation before replicator update, mutation every step) would give different dynamics.

### 14. The replicator's `agentprofiles` array is rebuilt as a sorted/contiguous structure

After `genBSreplicator_single_tstep`, `agentprofiles` is rebuilt as an ordered array where agents of the same type holding the same profile appear consecutively. This means agent index `i` does not correspond to a stable individual across timesteps — agent identity is essentially erased. This doesn't matter for the population-level dynamics, but it would matter if any analysis tried to track individuals over time. The paper's framing of "agents" as discrete entities implies stable identity that the code does not actually preserve.

---

## Things I Got Right but Could Have Explained More Deeply

These weren't oversights exactly, but my treatment was thinner than it should have been:

- The forgetting-before-reinforcement ordering: I noted it but didn't explain its consequence (old urn weights are decayed each timestep regardless of whether this timestep's interactions reinforce them, creating a constant memory-half-life independent of interaction frequency).

- The pre-batched `picks10` and `mutations10`: I called these performance optimizations but didn't note that they constrain randomness within a 10-timestep batch — the random draws for timesteps t through t+9 are committed simultaneously, which is functionally equivalent to per-timestep draws given that each batch uses different rows, but it's a different RNG-consumption pattern than naive implementation.

- The "exec" assortment variant counting only non-null matching dimensions: I described what it does but not why this matters. Two agents both broadcasting null in a dimension are not actively choosing to coordinate via that dimension's signal — they're both abstaining from it. Counting that as a match would inflate assortment for agents who happen to share an absence of signaling, which doesn't fit the intuition of homophily. The paper doesn't make this distinction explicit; the code disambiguates by using the exec variant.

---

## What This Means for the Definitive Mapping

The mapping is still substantially correct, but Part 7 ("Implementation Details Not Stated in the Paper") was significantly less comprehensive than it should have been. A genuinely complete version would add at least these items as their own entries: signal cost integrated into reinforcement; the Shannon MI vs synergistic information substitution (or flag the potential discrepancy); the cross-run relabeling consequence for figure interpretability; the strict convergence criterion; the non-negativity clamp; the consequences of `flip7zero_correction`; the `execurnsf` staleness in the replicator path; the argmax tie-breaking bias; and the timestep-ordering choices in mutation and the replicator update.

The most important of these — the one I think deserves attention if anyone is trying to reproduce the paper's results — is the Shannon MI vs synergistic information question. The paper makes specific quantitative claims about synergistic information being ~49% of total information in the intersectional equilibrium, but the analysis code in these scripts computes only Shannon mutual information. That gap needs to be resolved by either locating the PID analysis code or confirming that the paper's claim comes from external analysis.









**Follow-up prompt:** Does the mapping overlook anything specifically related to avoiding statistical biases in the replicator dynamics?



**Opus 4.7 response:**

Yes — this is a real oversight in the definitive mapping, and looking back at it carefully, I think I significantly underplayed how much of the replicator code is specifically engineered to avoid statistical biases that would otherwise compound over the simulation's 40,000 timesteps. The single line I gave cascade rounding doesn't do justice to what's actually happening.

## The Bias-Avoidance Architecture I Underplayed

### Cascade rounding is more than "preserves totals" — it's a designed bias-avoidance algorithm

I described cascade rounding in Part 7 as: "Converts float profile counts to integers via cumulative `np.around`, preserving type totals exactly. Prevents population drift."

This treats it like a routine numerical bookkeeping step. It's actually solving a non-trivial statistical problem. Consider the alternatives a less careful implementation might use:

- **Independent rounding** (round each profile's count separately): Each profile gets `round(count_x)`. The sum of rounded counts will generally not equal `typestotal[k]`, because some profiles round up and others round down without coordination. Over 40,000 timesteps, this produces a slow random walk in total population size — a substantial drift bias.

- **Truncation** (floor each count): Systematically rounds down, biasing the population *toward extinction*. Every step loses a small amount of total mass, and rare profiles (count < 1) are permanently eliminated even if they would have non-trivial cumulative contribution.

- **Floor with largest-remainder distribution**: Floor each count, then distribute remainders to whichever profiles have the largest fractional parts. This is unbiased in the aggregate but creates a deterministic preference for profiles whose fractional remainders are systematically high — and which profiles those are can be correlated with profile content in subtle ways.

The cascade method walks through profiles accumulating the cumulative float total, then rounds the cumulative sum at each step. The agents assigned to profile *i* equal `round(cumsum_through_i) − round(cumsum_through_i−1)`. This:

1. **Guarantees the total exactly equals `typestotal[k]`** because the final cumulative sum equals that total (post-renormalization).
2. **Allows profiles with count < 1 to occasionally receive an agent** when their contribution pushes the cumulative total across an integer boundary. With truncation, rare strategies would be permanently lost from the population each step.
3. **Distributes rounding errors evenly along the profile sequence** rather than concentrating them at high-fitness or low-fitness profiles.
4. **Uses `np.around` (banker's rounding — round-half-to-even)** rather than always-round-up or always-round-down, so the rounding direction is unbiased in expectation.

This last point — banker's rounding — I didn't mention at all. It's a statistically meaningful choice: standard "round half up" would systematically inflate boundary values, banker's rounding does not.

### The non-negativity clamp and renormalization together are a bias-avoidance compound

I noted the clamp exists. What I didn't articulate is the bias problem the clamp-then-renormalize sequence is solving.

The replicator equation `N_{t+1}(x) = N_t(x) + N_t(x) × (U − Avg)` can produce negative `N_{t+1}` when `(U − Avg) < −1` — that is, when a profile's utility is more than the average below it (rare but possible, especially with assortment weighting). If you renormalized *without* clamping first, the negative values would distort the renormalization factor in a way that doesn't correspond to any meaningful population dynamic. The resulting positive counts would be systematically inflated.

Clamping first means renormalization operates on a non-negative distribution. The mass lost to the clamp is then redistributed *proportionally to current counts*, which is replicator-consistent (it doesn't favor any specific profile other than via current frequency). The combined effect is to remove the bias from negative arithmetic while preserving the replicator's "rich get richer" character.

The specific ordering — **equation → clamp → renormalize → cascade round** — matters. Any other ordering produces different statistical properties.

### `flip7zero_correction` solves a population-extinction bias

I framed this as a "numerical safety function" for the rare zero-count event. But it's also a deliberate bias-avoidance mechanism. Without it, the first time cascade rounding drove any type to zero agents, that type would be **permanently eliminated** — and the model's premise of three coexisting preference types would be violated. Over a long simulation, the probability of at least one type going to zero at least once accumulates, so the absence of this correction would bias the long-run results toward fewer-type configurations.

The choice of *uniform reinitialization* (rather than, say, reinitializing the type with their last-known dominant profile) is itself an interesting choice: it sacrifices learned history for a clean reset, which prevents the corrected type from immediately re-collapsing if their previous distribution was already heading toward zero.

---

## Bias-Avoidance Mechanisms I Missed Entirely

### Self-pairing correction with M(i) − 1

I noted this is paper-specified and that the code implements it. What I didn't articulate is *why this matters as bias avoidance*. Without the M(i) − 1 correction, profiles that are common in the population would get "self-interaction" credit they don't deserve — an agent would effectively count itself among its interaction partners. This biases the utility calculation toward common strategies (a self-reinforcing effect), since common profiles would appear to coordinate well with themselves more than they actually do. The −1 correction prevents this self-counting bias.

### Type totals are preserved as constants, not derived

`typestotal[k]` is computed once at initialization and held constant. The renormalization in every timestep rescales `profile_count_typed[k]` to sum to exactly this constant. If type totals were allowed to drift (e.g., if you trusted the replicator equation to preserve them in floating-point arithmetic), small numerical errors would accumulate over 40,000 timesteps and produce population drift. By making the constant explicit and forcing renormalization to it every step, this drift is prevented.

This is a different bias than cascade rounding addresses (cascade rounding handles the float-to-int conversion bias; the typestotal constant handles the floating-point arithmetic drift bias). Both are needed.

### Mutation timing within the replicator step

The code applies mutation **after** the replicator update each timestep, not before. The paper doesn't specify ordering. The choice has bias implications:

- **Mutation after replicator update**: The replicator equation operates on the post-mutation distribution from the previous step, propagates it forward, then mutations perturb the result. Mutations occur in proportion to current population frequencies (a mutating Lagashite is more likely than a mutating Akkadian if Lagashites outnumber Akkadians).

- **Mutation before replicator update** (alternative): The mutations perturb the distribution, then the replicator equation amplifies whichever mutated profiles happen to have above-average utility. This would amplify the impact of fortunate mutations more strongly.

The chosen ordering is more conservative — mutations are tested by the next utility calculation before being amplified. This reduces the impact of any single mutation event on long-term trajectories.

### The batched mutation RNG doesn't introduce temporal correlation

The `picks10` / `mutations10` / `muteprofilestrings10` batches are regenerated every 10 timesteps. A reader might worry this introduces 10-timestep temporal correlation. It doesn't, because different rows of the batch are used for different timesteps — they're independently drawn within the batch. The batching is purely a performance optimization to avoid repeated RNG calls inside numba; it has no statistical consequences. But this is non-obvious from inspection and worth confirming in any rigorous bias audit.

---

## A Bias the Code Does NOT Avoid (And Probably Should Not)

### Cascade rounding's profile-order dependence

The cascade rounding walks through profile indices in their natural order (0, 1, 2, ..., numprofiles−1). When the cumulative float total falls between two integers, the agent assignment depends on which profile happens to be the one that pushes the cumulative sum across the next integer boundary. Profiles with lower indices encounter the boundary-crossing opportunities first.

In principle, this introduces a slight systematic preference for low-index profiles to receive the "fractional excess" agents. Whether this matters depends on whether profile index correlates with anything meaningful. In this codebase, `profiles_indexed` is built by lexicographic enumeration of signal-then-action combinations, so low-index profiles tend to be those with null signals in early dimensions and specific action choices. There's no reason to expect this index ordering to correlate with utility in any systematic way, so the bias likely averages out over many timesteps.

But the code does not actively randomize the profile iteration order to *guarantee* this bias averages out. A more paranoid implementation would shuffle the profile order before cascade rounding. The code accepts this tiny residual bias as the price of determinism and performance.

---

## What the Definitive Mapping Should Have Said

The Part 7 table should have included a dedicated subsection on the replicator's bias-avoidance architecture, treating cascade rounding, the clamp-renormalize-round ordering, `flip7zero_correction`, the constant `typestotal`, and `np.around`'s banker's rounding as a *coordinated set* of mechanisms designed to prevent specific bias modes (population drift, profile-favoritism, rare-strategy elimination, type extinction, sign error inflation). Treating them as separate one-line entries — which is what I did — misses the design coherence and makes the implementation look like a collection of ad-hoc safeguards rather than what it actually is: a carefully constructed numerical realization of the replicator equation that preserves the model's statistical properties over long simulation horizons.

The single most important addition would be a sentence acknowledging that **the deterministic replicator equation in continuous form is unbiased by construction**, but **discretizing it into integer agent populations is not unbiased by default** — the bias-avoidance work is required precisely because the simulation operates on discrete agents rather than continuous frequencies, and this is a substantive implementation contribution the paper does not need to (and does not) describe.
