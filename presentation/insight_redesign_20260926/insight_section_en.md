# From Client Allocation to Functional Support and Retention

We examine three aspects of shared-model adaptation: client allocation, the sources of support for individual classification functions, and the persistence of previously attained improvements. Test accuracy and training-witness diagnostics serve different purposes and are reported separately.

## Tail retention under fixed global class counts

We compare Client-LT and a standard Dirichlet allocation of the same CIFAR100-LT training pool. Both allocations contain the same 10,847 images and global class counts, use 30 clients, and follow the same logit-adjustment objective and staged update schedule. In the current single-seed experiment, their mean overall accuracies over rounds 81–100 are 70.910% and 71.014%, respectively. Despite this 0.104 percentage-point difference, their tail accuracies are 68.663% and 72.298%, a gap of 3.635 percentage points. Improvements on non-tail classes partly offset the tail deficit in the overall metric.

The trajectories also distinguish attained accuracy from subsequent retention. Under Client-LT, tail accuracy falls from a peak of 71.80% to 68.35% at round 100. Under the Dirichlet allocation, it falls from 72.90% to 72.00%. Thus, fixed global class counts can accompany different retention trajectories under the evaluated allocation protocols. The allocations also differ in client capacity, class coverage, and local class mixtures, with slightly different realized optimization-step counts. This comparison therefore does not isolate any single property of the client–class structure.

These observations motivate a closer examination of how client updates support particular classification functions during shared adaptation.

## Label ownership and functional support

We define a functional unit as a fixed set of training-witness images for a class on a particular client. Its score is the mean cosine-similarity margin between the correct class and the strongest competing class. At a common model state after B aggregation and before the A update, we estimate a client proposal's effect using its inner product with the functional gradient. A source is counted as positive only when the smaller response across two fixed views exceeds 10⁻⁶. This measures potential functional support, whereas label ownership describes where training examples reside.

We audit all 90 A-update rounds of the current Full-CP trajectory. Averaging functional units within classes, followed by equal averaging across eligible classes and rounds, 98.65% of supported tail units have at least one positive source that does not own the target label. Units without positive support, and classes without any supported unit in a given round, are excluded from this conditional statistic. On this protected trajectory, restricting source measurement to label owners would therefore omit potential positive support.

This observation does not establish that an individual non-owner is more helpful. Under the same conditional averaging, non-owners constitute 87.81% of candidate sources for tail units and account for 86.46% of positive response magnitude. The responses are also unweighted by aggregation coefficients and do not measure realized contributions to the shared model. We use functional responses to characterize support and treat concentration-based prioritization as an optimization choice to be evaluated through controlled substitutions, rather than as a consequence of the source-occurrence statistic.

## Current improvement and persistent retention

Positive support in a given round does not itself ensure that earlier improvements persist. We distinguish the common starting state, the ordinary A proposal, the committed model, and the state after the next B aggregation. Current targets seek improvements supported by this round's proposals. Historical references instead record functional levels sustained by previously committed models, using only information available at the time of registration.

An earlier experiment without the classification-preservation regularizer provides preliminary evidence for this distinction. With the same retention coefficient, using only the current target yields a mean tail accuracy of 68.963% and a peak-to-final drop of 3.25 percentage points. Combining current and historical targets yields 71.073% and a drop of 0.70 percentage points, while overall accuracy decreases from 70.861% to 70.565%. This conditional comparison supports examining persistent retention separately from current improvement, while exposing an adaptation trade-off. It does not replace the corresponding ablation in the current CP version.

These observations motivate combining response-based improvement targets with previously sustained functional levels during shared adaptation. Whether source concentration independently predicts future deterioration requires a separate temporal analysis on a trajectory not already influenced by source-aware protection. We do not assume that causal relationship in the present design.

---

Draft note: These paragraphs report seed42 results and distinguish the current Full-CP diagnostic from the earlier non-CP history ablation. Accuracy summaries use rounds 81–100; peak-to-final drops use round 100. Local data sources and experimental boundaries are recorded separately in `证据与实验清单.md`.
