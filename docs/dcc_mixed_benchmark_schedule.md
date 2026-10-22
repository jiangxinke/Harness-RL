# Scheduling mixed CUAD and Web training

The online trainer defaults to `--mixed-benchmark-schedule slot-interleaved`.
It repeats the CUAD/Web mixture across flattened task slots and then splits
those slots into global iterations. The default plan and configuration identity
are preserved: omitting the option or setting it explicitly to `slot-interleaved`
adds no scheduling field to saved configuration. With a 1:1 mixture and an even
world size, the default assigns even ranks to CUAD and odd ranks to Web in every
iteration. Faster ranks wait for the slowest rank before the shared update.

To let every rank alternate benchmarks, add these options to an existing online
Web training command:

```sh
--enable-online-web \
--cuad-draws-per-cycle 1 --web-draws-per-cycle 1 \
--steps 4 --mixed-benchmark-schedule iteration-blocked
```

In this mode each global iteration contains one benchmark on every rank.
The example schedules CUAD, Web, CUAD, Web, giving each rank two groups from
each benchmark. For world size W, it retains 4W groups, with 2W per benchmark.
At group size four, that is 16W episode attempts, 8W per benchmark. These are
draw counts; tasks can repeat when a deck cycles. Other positive mixtures work
the same way: a 3:1 cycle contains three all-CUAD iterations followed by one
all-Web iteration. Every rank receives the same benchmark counts.

`iteration-blocked` requires `--enable-online-web` and a positive `--steps`
that is an integer multiple of
`cuad_draws_per_cycle + web_draws_per_cycle`. An incomplete cycle is rejected,
even when the total number of flattened slots would form a complete cycle.
Across complete cycles, the original per-benchmark global counts, CUAD sampling
mode, independent RNG/decks, draw order within each deck and task multiset are
preserved. No reward, group size, gradient averaging or failure handling changes.
Configuration records the selected mode and that optimization order changed.

This changes the composition and order of optimizer updates, task-to-rank RNG
assignment and the policy snapshots used for subsequent sampling. It is a new
training schedule. Results cannot be joined to an old control and presented as
a matched A/B gain; a fresh comparison must disclose the schedule.

Homogeneous iterations can reduce systematic waiting when Web groups take
longer than CUAD. They do not guarantee full GPU utilization or a speedup:
network waits remain, Web task lengths vary, and an all-Web iteration increases
peak Web concurrency. The existing finite full-world readiness barrier remains.
Episode deadlines include queue waiting, but safe completion of in-flight
native requests and replay can extend beyond a deadline. Four concurrently
started episodes with 1800-second deadlines do not establish a hard 7200-second
group bound. Changing scheduling does not change native batch size or admit a
particular memory configuration.
