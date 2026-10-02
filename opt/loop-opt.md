# Optimizing loops
Loops are said to contain 80% of computations in 20% of the code.
Because they describe repetitive work that can be executed independently,
they are often excellent candidates for vectorization and multi-threading.
In this section, we discuss strategies to enable loop parallelism using the
Ripple compiler.

Loop parallelization consists of:
- finding loops whose iterations can be mapped to a form of parallelism
(SIMD, instruction-level parallelism, thread-level parallelism).
- Identify the form of parallelism that corresponds best to the loop
- Transforming the loop into a parallel loop.

Here we go through to the different forms of parallelism and how to detect that
a loop is a good candidate for it.

## SIMD parallelism
- Only parallel loops, i.e., in which iterations are independent from each other,
  are eligible
- Perhaps the most important performance factor for vector parallelism is that
  memory accesses are [coalesced](./coalescing.md).

A good strategy is to look at how memory accesses depend upon the counter
(let's call it `i`) associated with the candidate loop.
If:
- all the accesses that depend upon `i` grow by one element in memory
as `i` grows by 1, 
- there are enough values of `i` to utilize the targeted SIMD width,

then loop `i` is a great candidate for SIMD vectorization, which can be done
either by calling `ripple_parallel()` before loop `i`, or by using the SPMD form.

## Instruction-level parallelism
Modern processors often leverage forms of parallelism that allow the execution
of different instructions to overlap in time.
Examples of this are:
- VLIW processors, in which instructions are organized in fixed-sized packets.
Instructions in a packet are typically executed simultaneously
by different ALUs in the processor.
Optimizing for VLIW is about increasing the utilization of these ALUs.
- pipelined processors, in which the execution of instructions is split up
in small steps.
Optimizing for pipelined processors is about maximizing the use of
all these steps.

A typical issue occurs when consecutive instructions depend upon each other,
(say `b` depends upon `a`) and when the latency of `a` is longer than a cycle.
The processor needs to wait, and `b` cannot be executed at the next cycle
(i.e. the next VLIW packet, or the next pipeline stage).
The processor needs to wait, which is inefficient.

The typical way to solve these delays ("pipeline stalls", "pipeline bubbles"),
is to provide instructions that are independent from `a` and `b`,
which can be executed between `a` and `b`, filling up the delay / gap
between `a` and `b`.

In loop codes, these independent instructions are found in adjacent iterations
of a loop.
The loop may be parallel (independent iterations), in which case
all the instructions in the next iteration are independent from
the iterations in the current iteration.
But it may also not be parallel, in which case
some instructions in the next iteration may still be independent from
the instructions in the current iteration.
Both cases can allow the compiler to fill out the gap.

### Loop unrolling
It is possible to expose this extra parallelism by "unrolling" a loop,
i.e., by explicitly writing its body `n` times, as in the example below.
`n` is called the "unroll factor".

```C
float A[64];
for (unsigned i =0 ; i < 64; ++i) {
  A[i] = i * i;
}
```
Unrolling the `i` loop with a factor of `2` becomes:
```C
for (unsigned i2 = 0; i2 < 32; i2++) {
  A[2 * i2] = (2 * i2) * (2 * i2);
  A[2 * i2 + 1] = (2 * i2 + 1) * (2 * i2 + 1);
}
```

### Using pragma to unroll loops
Unrolling loops by hand increases the code size, is error-prone, and reduces
the readability of the code.

`clang`, Ripple's underlying C/C++ compiler,
offers pragma annotations to unroll loops for you,
leaving the original loop body untouched.

The unrolled `i2` loop above can be obtained by annotating loop `i` as such:

```C
float A[64];
#pragma clang loop unroll_count(2)
for (unsigned i =0 ; i < 64; ++i) {
  A[i] = i * i;
}
```

### Using Ripple to increase the computational density of code
By doubling the size of the Ripple block, we specify that twice the amount of
work needs to be done by each instruction associated with the block.
This results in a form of instruction-level parallelism that is similar
to unrolling a parallel loop, except that it makes it easier for the compiler
to exploit instruction-level parallelism.

In our running example, let's assume that we have vectorized `i`
on a block of 32 elements:
```C
ripple_block_t block = ripple_set_block_shape(VECTOR_PE, 32);
ripple_parallel_full(block, 0); // no epilogue in this case -> _full
for (unsigned i = 0; i < 64; ++i) {
  A[i] = i * i;
}
```
We can double the instruction-level parallelism by doubling the block size:

```C
ripple_block_t block = ripple_set_block_shape(VECTOR_PE, 64);
ripple_parallel_full(block, 0);
for (unsigned i = 0; i < 64; ++i) {
  A[i] = i * i;
}
```
#### Do not unroll a loop also vectorized with Ripple
__Performance Impact__: Medium to High.
Looking at the `i2` loop above, we see that accesses to `A` now
increase by 2 elements when we increase `i2` by 1.
As a result, using the `i2` loop for vectorization would produce uncoalesced
memory accesses (with a stride of 2 elements).

### Tradeoffs
__Performance Impact__: Medium to High.
Optimal instruction-level parallelism boils down to choosing
the right value for `n`.
- Unrolling is only useful to fill out a lack of instruction-level parallelism.
If no pipeline / VLIW delays are present, there is no need to use loop unrolling.
- Interleaving more independent instructions results in higher register pressure.
Requiring more registers than available on the targeted processor
results in stack spilling, which significantly lowers performance.
- Growing `n` also results in bigger code sizes,
and often longer compilation times.

In short, an optimal unroll factor is one that inserts independent instructions
that can fill pipeline stalls and VLIW slots,
without requiring more registers than the register file provides.

## Multi-thread parallelism (cf. Ripple Manual)
Multi-threading is available through loop annotations using variants of `ripple_thd_parallel()`, or SPMD by directly using `ripple_thd_id()`.
Only a parallel loop can be used for multi-threading.
The choice of a loop is typically driven by data locality associated with it.
Threads sharing a cache will want to have a smaller collective data footprint
than threads that don't.
The way the processor's hardware schedules threads also has an important impact.
This topic is discussed further in the target-specific parts of
this optimization guide.
