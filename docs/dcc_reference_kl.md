# Optional frozen-base reference regularization

`scripts/dcc_online_grpo.py --reference-kl-beta 0` (the default) keeps the
original GRPO objective and model-scoring call count. A finite positive beta
opts into the separately reported
`frozen-base-reference-k3-nonzero-advantage-thought-v1` algorithm variant.
This is an experimental regularizer, not a known bug fix or a claim of better
benchmark results.

For each thought segment already scored by GRPO (its episode advantage must
be nonzero), the trainer first scores the exact thought IDs under the same
loaded, frozen base using PEFT `disable_adapter()` and `torch.no_grad()`.
The reference uses the training base's actual precision and quantization.
It does not use a warm-start adapter as reference. Only one standard,
uniformly enabled, unmerged q/v/o LoRA adapter with all its parameters trainable
and no trained base biases is supported. Partial adapter freezing is rejected
because PEFT adapter restoration can otherwise change parameter trainability.

The adapter context and reference eval mode finish before the differentiable
current-policy forward. Each segment's backward completes before another
reference read, so checkpoint recomputation observes the restored adapter.
The scorer must not be used concurrently with a reference read.

The added term is

```text
beta * sum(expm1(ref_logp - current_logp) - (ref_logp - current_logp))
    / (group_size * max_steps * thought_tokens)
```

This is a sampled-token K3 regularizer, not an exact vocabulary-wide KL.
It uses the existing sampled behavior trajectories; it adds no importance
correction to the KL term. The policy surrogate still uses actual sampled
behavior logprobs as its old-policy denominator. JSON tokens, zero-advantage
episodes, and zero-signal groups receive no added loss. Advantage normalization,
fixed budgets, gradient averaging, clipping, and zero-global-gradient optimizer
skipping retain their original behavior.

Configuration records beta, base/source identity, reference method, active
scope, and algorithm variant. Step reports separate `policy_loss`,
`reference_kl_unweighted_loss`, `reference_kl_loss` (already multiplied by beta),
and `objective_loss`. An active segment costs one additional base forward.
`expm1` reduces cancellation near zero; nonfinite K3/objective values are
rejected before an update instead of silently clipping the objective. Large
drift can still cause overflow or high-variance gradients. A prospective TRAIN
comparison is needed before adopting any positive beta.
