# Multi-threading in Ripple
Ripple allows you to use the SPMD and loop annotation parallel
programming models to express multi-threaded
(sometimes also called "self-threaded") programs.

## Ripple threading API
The Ripple thread API is prefixed with `ripple_thd`.
Just like for SIMD, we can declare a block (of threads),
retrieve the size of the block along any dimension,
and get the index of your current thread along any dimension.
We can also annotate loops to distribute their iterations along one or more
loops using the `ripple_thd_parallel` API,
which supports static and dynamic scheduling.

Similar to the SIMD programming model,
the loop annotation style is syntactic sugar provided by Ripple
on top of SPMD. As a result, both styles can be mixed if necessary.

### Binding Ripple to a threading runtime
Ripple thread block objects communicate with an underlying threading runtime,
through a runtime object representing said underlying runtime.
To declare a thread block in Ripple, we need to initialize the underlying
runtime object,
and then create a Ripple thread block object using the runtime object.

The code below can be found in `ripple/thread.h`.

```C
// @brief Initializes @n_blocks ripple_thd_block_t objects
/// from a pointer to the @p underlying runtime object.
/// @param flags enables features that we want to use,
/// e.g RIPPLE_THD_USE_BAR for barriers,
/// and RIPPLE_THD_USE_DYN for dynamic load balancing
/// @param max_dims the maximum number of thread dimensions to use.
/// This impacts the size of the runtime objects.
void ripple_thd_init(int PE_id, void * underlying,
                     unsigned n_blocks,
                     unsigned flags,
                     unsigned max_dims);

// @brief Tears down the Ripple runtime objects associated with the underlying
/// runtime object @p underlying.
void ripple_thd_exit(void * underlying);
```

The `flags` parameter defines the special multi-threading features we intend
to use (barriers, dynamic load balancing).
Not using them saves some init time and synchronization object space.

```C
// _____ Flags for Ripple runtime initialization _____

/// Whether we intend to use dynamic load balancing (ripple_parallel_dyn)
#define RIPPLE_THD_USE_DYN 1u
/// Whether we intend to use barriers
#define RIPPLE_THD_USE_BAR (1u << 1)
/// Default flags for ripple_thd_init
#define RIPPLE_THD_DFT (RIPPLE_THD_USE_DYN | RIPPLE_THD_USE_BAR)
/// Whether we don't intend to use any runtime helpers (load-balancing,
/// barriers)
#define RIPPLE_THD_NONE 0
```

`max_dims` is currently 3 for all runtimes, and `n_blocks` can only be `1`
in the QHPI underlying runtime.

For each supported underlying threading runtime,
Ripple uses a different type of underlying threading runtime object.
These can also be used to save boilerplate thread initialization and
termination code.
Supported runtimes and their underlying runtime are represented
in the following table:

| Runtime | Underlying Type | Include `ripple/thread.h` and link with |
|---------|----------------|-----------|
| QuRT    | `qthd_runtime_t` | `-lripple_thd_qurt` |
| QHPI    | `QHPI_RuntimeHandle` | `-lripple_thd_qhpi` |

How underlying runtimes are initialized, and how the threads are spawned,
are defined on a per-runtime basis.
For instance, QHPI itself offers an environment in which the threads are
already running, and the underlying runtime object is already initialized.
Conversely, the underlying runtime object in QuRT needs to be initialized before
spawning the threads.

We will see how to create an underlying runtime object and spawn threads
for [QuRT](#qurt-specific-api) and [QHPI](#qhpi-specific-api) below.

### SPMD API
```C
#include <ripple/thread.h>

/// \brief Defines a thread block shape for the threads provided through
/// the \p underlying_runtime threading runtime object.
/// \p block_id when the underlying runtime supports multiple thread blocks,
///             this is the index of the block to use.
/// \param n_dims the number of dimensions of the thread block shape
/// \param shape the thread block shape.
/// A special constant, `RIPPLE_THD_DYNAMIC` can be used in one of the dimensions
/// to mean "as many as the number of underlying threads allow".
/// Currently, for `RIPPLE_THD_DYNAMIC` to work,
/// there must exist a value for which the volume of the resulting
/// shape matches the number of spawned threads exactly
/// (which is always true for one-dimensional shapes)
ripple_thd_block_t ripple_thd_set_block_shape(void *underlying_runtime,
                                              unsigned block_id,
                                              unsigned n_dims, size_t ... shape);

/// \brief Index of the current thread along dimension \p dim.
size_t ripple_thd_id(ripple_thd_block_t b, unsigned dim);

/// \brief Size of thread block \p b along dimension \p dim.
size_t ripple_thd_get_block_size(ripple_thd_block_t b, unsigned dim);


/// \brief an inter-thread barrier.
/// \param dims This is a future parameter to select which
///        dimensions are synchronizing through a barrier.
///        Currently, this is an all-thread barrier (dim = -1).
void ripple_thd_barrier(ripple_thd_block_t b, unsigned dims);

/// \brief Whether the indices of the current thread along dimensions \p dims
///        are zero.
/// \param dims A bitset representing the thread block dimensions of interest.
///             Can be -1 for "all".
int ripple_thd_is_main(ripple_thd_block_t b, unsigned dims);
```

### Loop annotations API
```C
#include <ripple/thread.h>

/// \brief Tells Ripple to statically distribute the iterations of the for loop
/// immediately following this call.
/// Chunks of \p chunk_size contiguous iterations get distributed cyclically
/// across dimension \p dims of the thread block.
/// \param flags future parameter that will toggle behaviors like an implicit
///        barrier at the end of the loop.
///        Default flag is RIPPLE_THD_PAR_DFT.
///
/// A static distribution determines a fixed mapping from iteration chunks
/// to threads. It has virtually no runtime overhead and its per-thread
/// execution is deterministic. However, it can suffer from load imbalance
/// when the amount of work in the chunks is too heterogeneous.
void ripple_thd_parallel(ripple_thd_block_t b, unsigned chunk_size, int flags, unsigned ... dims);

/// \brief Tells Ripple to dynamically distribute the iterations of the for loop
/// immediately following this call.
/// Chunks of \p chunk_size iterations get distributed dynamically across
/// dimension \p dims of the thread block.
/// Only dimension 0 is supported.
//  This is enough to express all degrees of dynamic parallelism necessary.
/// \param flags future parameter that will toggle behaviors like an implicit
///        barrier at the end of the loop.
///        Default flag is RIPPLE_THD_PAR_DFT.
///
/// A dynamic distribution lets threads process a chunk
/// whenever they are done with the previous one, making the execution tolerant
/// to heterogeneous chunks (i.e, more load-balanced).
/// The downside of dynamic distributions is that
/// they come with some runtime overhead.
void ripple_thd_parallel_dyn(ripple_thd_block_t b, unsigned chunk_size, int flags, unsigned ... dims);
```


### QuRT-specific API
While the API above is universal across runtimes,
we rely on a few runtime-specific APIs to make the use of multi-threaded Ripple 
easier for some runtimes.

```C
/// \brief Synchronously calls \p func with \p args from all threads.
void ripple_thd_call(ripple_thd_block_t b, void *(*func)(void *), void *args);
```

The QuRT-specific runtime object, which represents QuRT to Ripple,
is created and destroyed using the following API:
```C
/// @brief Creates a qthread-based environment that runs @p n_thd threads,
/// Which can be used to create a Ripple block.
extern qthd_runtime_t *qthd_runtime_init(unsigned n_thd);

/// @brief Tears down the qthread-based environment
extern void qthd_runtime_exit(qthd_runtime_t *rt);
```

To create a multi-threaded function with the SPMD model,
we need a thread block object.
We can then access the SPMD thread indices and barriers from that block.

In Ripple threads, the thread block is a runtime object,
which can be passed around among functions.

The following QuRT-based example code implements a simple multi-threaded,
non-SIMD vector addition.
The work is split among equally-sized chunks of contiguous iterations.
Each thread executes one chunk.
Here we're using an `if` statement to prevent array overflow.
A more efficient way to obtain that would be to separate the last-thread case
from the other ones, because it executes fewer iterations.
By doing said separation, the `if` would move out of the loop,
reducing the cost of control during the loop execution.

```C
#include <stddef.h>
#include <ripple/thread.h>
#include <ripple/ripple_thd_qurt.h>
#define NUM_THREADS 16
#define THREADS 0
#define N 1024
// We make the underlying runtime object accessible globally
qthd_runtime_t *rt;

float A[N], B[N], SUM[N];

typedef struct {
  size_t n;
  float * a;
  float * b;
  float * sum;
} args_t;

// QuRT threads work similarly to pthreads:
// entry functions need to unpack arguments from a struct.
void * vecadd(void * vargs) {
  args_t * args = (args_t *)vargs;
  size_t n = args->n;
  float * a = args->a;
  float * b = args->b;
  float * sum = args->sum;
  // By using `RIPPLE_THD_DYNAMIC`, we use all the threads available (i.e. 16).
  ripple_thd_block_t thdb =
    ripple_thd_set_block_shape(rt, /*block*/0, /*n_dims*/1, RIPPLE_THD_DYNAMIC);
  size_t thd_id = ripple_thd_id(thdb, /*dimension*/ 0);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  // This chunk size evenly distributes one chunk per thread
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  for (size_t i = thd_id * chunk_size; i < (thd_id + 1) * chunk_size; ++i) {
    if (i < n)
      sum[i] = a[i] + b[i];
  }
}

int main() {
  rt = qthd_runtime_init(NUM_THREADS);
  // In QuRT, Ripple thread runtime is bound to Ripple blocks
  // BEFORE spawning the threads.
  // We're only planning to use one-dimensional thread blocks.
  ripple_thd_block_t b = ripple_thd_init(THREADS, rt, /*n_blocks*/1,
                                         /*flags*/RIPPLE_THD_DFT,
                                         /*max_dims*/1);
  args_t args = {N, A, B, SUM};
  // This API is runtime-specific (here it comes from ripple_thd_qurt.h)
  ripple_thd_call(b, vecadd, &args); // synchronous call, args doesn't escape
  qthd_runtime_exit(rt);
}
```

### QHPI-specific API

QHPI offers an ambient multi-threaded environment, and passes its runtime
object directly to the kernels.
Hence, the underlying runtime object needs to be bound to Ripple blocks
using `ripple_thd_init()` within said ambient multi-threaded environment.

The same SPMD vector addition kernel as before would look like below.

```C++
#include "HTP/core/qhpi.h"
#include <ripple/thread.h>
#include <ripple/ripple_thd_qhpi.h>
#define THREADS 0

uint32_t vecadd(QHPI_RuntimeHandle *rt,
                uint32_t n_outputs, QHPI_Tensor ** outputs,
                uint32_t n_inputs, const QHPI_Tensor *const * inputs) {

  // _________ Begin QHPI boilerplate parameter unpacking __________
  const float * a = (const float *) qhpi_tensor_raw_data(inputs[0]);
  const float * b = (const float *) qhpi_tensor_raw_data(inputs[1]);
  float * sum = (float *) qhpi_tensor_raw_data(outputs[0]);
  QHPI_Shape shape = qhpi_tensor_shape(outputs[0]);
  uint32_t n = shape_size(shape);
  // _______________ End QHPI parameter unpacking __________________

  // In QHPI, thread blocks are bound with the QHPI runtime here
  // (in QHPI's multi-threaded environment)
  ripple_thd_init(THREADS, rt, /*n_blocks*/1, /*flags*/RIPPLE_THD_DFT, /*max_dims*/1);
  ripple_thd_block_t thdb =
    ripple_thd_set_block_shape(rt, /*block*/0, /*n_dims*/1, RIPPLE_THD_DYNAMIC);
  size_t thd_id = ripple_thd_id(thdb, /*dimension*/ 0);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  for (size_t i = thd_id * chunk_size; i < (thd_id + 1) * chunk_size; ++i) {
    if (i < n)
      sum[i] = a[i] + b[i];
  }
  ripple_thd_exit(rt);
  return QHPI_SUCCESS;
}
```


## Loop annotation model for threads
Ripple loop annotations allow us to not worry about writing thread loop bounds,
conditionals, etc in a multi-threaded loop.
Instead, we can focus on how we want to distribute the work of a loop across
threads using only a few parameters:
- which loops should provide the thread parallelism
- grain of parallelism (chunk size) for each of these loops
- whether to use static or dynamic distribution of work among threads.

### Static distribution
The vecadd function above can be simplified _and improved_
by using ripple loop annotations, as follows:
```C
void * vecadd(void * vargs) {
  args_t * args = (args_t *)vargs;
  size_t n = args->n;
  float * a = args->a;
  float * b = args->b;
  float * sum = args->sum;
  // By using `RIPPLE_THD_DYNAMIC`, we use all the threads available (i.e. 16).
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*block*/0, /*n_dims*/1, /*shape*/RIPPLE_THD_DYNAMIC);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  ripple_thd_parallel(thdb, chunk_size, /*flags*/RIPPLE_THD_PAR_DFT, /*dims*/0);
  for (size_t i = 0; i < n; ++i) {
      sum[i] = a[i] + b[i];
  }
}
```
Ripple interprets the `ripple_thd_parallel()` API call, by refactoring the i
loop into a multi-threaded loop that assigns a contiguous block of `chunk_size`
iterations to each thread. Except for the last one if `n` is not a multiple of
`chunk_size`, in which case the last thread case is separated.

In this example, `chunk_size` is such that each thread only executes one chunk.
But in general, when chunk_size is smaller, chunks are distributed cyclically
(i.e. round-robin) across threads.

### Why chunk the iterations ?
Threading is a form of parallelism that tends to benefit from bulkier
computations.
The `chunk_size` parameter provides a simple way to create bulkier tasks
by allowing threads to execute a group of contiguous iterations,
rather than having to process a single iteration at a time.

In more rare cases, the work present in a single loop iteration is sufficient or
appropriate as a per-thread work unit.
In that case, we can use `chunk_size=1`.


### Dynamic distribution

Multi-threaded code execution is faster when each thread takes about the same
time to execute the code.
That way, threads don't have to wait for the completion of other threads.
We call this ideal situation a load-balanced execution.

When a loop's iterations perform identical calculations,
giving the same amount of iterations to all threads results in good
load balancing.

When the workload varies from one iteration to the next,
we need to split the iterations less regularly,
in order to balance the work across threads.
To enable such load balancing,
Ripple supports a generic solution based on _dynamic scheduling_ of chunks.

In dynamic scheduling, threads pick up a new available iteration chunk
as soon as they are done with the previous chunk.
The new iteration chunk is taken from the pool of all iteration chunks
that need to be executed by all the threads.
This multi-threaded scheduling method is often called "work sharing."

The following `matrix_trans()` example performs an in-place
matrix transposition, i.e. swaps the elements of
the lower-triangular part of `A` with their upper-triangular counterpart.

```C
void matrix_trans(size_t n, float A[n][n]) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      float tmp = A[i][j];
      A[i][j] = A[j][i];
      A[j][i] = tmp;
    }
  }
}
```

In `matrix_trans()`, `i` iterations of loop `j` are executed within
each iteration `i`: when `i` is small, loop `j` does a small amount of work,
and when `i` gets bigger, loop `j` does a bigger amount of work.
This means that work is not homogeneous across loop `i`.
If we would distribute loop `i` among threads using static scheduling,
say across 4 threads, the amount of work assigned to thread 3
would be considerably larger than what's assigned to thread 2, 1, and 0.
Dynamic scheduling provides better load balancing.
The thread-parallel, dynamically-scheduled version of `matrix_trans()`
is represented below.

```C
qthd_runtime_t * rt; // Assumed to be initialized elsewhere

// We assume that matrix_trans is called from all threads
void matrix_trans(size_t n, float A[n][n]) {
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*block*/0, /*n_dims*/1, RIPPLE_THD_DYNAMIC);
  ripple_thd_parallel_dyn(thdb, /*chunk_size*/16, /*flags*/RIPPLE_THD_PAR_DFT, /*dims*/0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      float tmp = A[i][j];
      A[i][j] = A[j][i];
      A[j][i] = tmp;
    }
  }
}
```


### How to determine chunk sizes
Several factors determine the optimal chunk size:
- __Data locality__: often, data is accessed such that contiguous iterations of
a certain loop will either reuse some data (temporal locality),
or will access data whose addresses are close (spatial locality).
These types of accesses are notoriously good for performance
as they tend to enable better use of caches, DMAs,
and other hardware-accelerating features.
- __Alignment__: Chunk sizes will determine the start address of data
accessed by each chunk.
Choosing a chunk size that correspond to hardware alignment constraints
can enable aligned operations.
- __Control overhead__: small chunks come with more control overhead,
because they have to "jump" to the beginning of a loop more often.
When using dynamic parallelism, switching from one chunk (task) to the next
can come with non-trivial synchronization costs.
Reducing the number of such switchings is achieved through bigger chunks.
- __Thread utilization__: When using static scheduling,
it is often beneficial to ensure that the work is spread along all threads.
Too-big chunks could create a shortage of chunks, i.e. thread under-utilization.
- __Load balancing__: when using dynamic scheduling, there needs to be
enough chunks to create an opportunity for load-balancing.
For instance, if there is only one chunk per thread,
there is no way to balance the chunks across threads
while still utilizing all threads.
More opportunities arise as the number of chunks per thread grows.

The following table summarizes these main factors.

| Factor | Optimal Chunk Size |
|--------|-------------------|
| **Data locality** | Larger chunks → better cache/DMA reuse (temporal + spatial locality) |
| **Alignment** | Chunks are multiple of alignment sizes → more aligned load/stores |
| **Control overhead** | Larger chunks → fewer loop jumps and synchronization costs |
| **Thread utilization** (static) | Smaller chunks → more chunks → better spread across threads |
| **Load balancing** (dynamic) | Smaller chunks → more chunks per thread → more balancing opportunities |

### Hybrid scheduling
Since Ripple supports multi-dimensional thread blocks,
there is an opportunity to trade off between static and dynamic scheduling, by using static scheduling for some thread dimensions,
and dynamic scheduling for others.

__Constraints: there is a maximum of one dynamically-scheduled dimension, and that dimension has to be dimension 0__.
This doesn't mean that the dynamically scheduled dimension has to be
the innermost loop.
There is no ordering constraints among dimensions of a thread block.

The following table illustrates the various combinations in the case of
a two-dimensional thread block:

| outer loop | inner loop | Dynamicity |
|------------|------------|------------|
|  static    | static     | pure static|
|  dynamic   | static     | coarse-grain dynamic: balancing statically-scheduled sets of chunks |
| static | dynamic | fine-grain dynamic: load-balancing among subsets of threads|
| dynamic | dynamic | not supported |

The outer-dynamic, inner-static option offers an interesting compromise
between load balancing and runtime overhead.
The outer-static, inner-dynamic option is suitable for cases when we know that
the inner loops perform heterogeneous work,
but the sum of this heterogeneous work is roughly similar across chunks
of the outer loop.
This is likely a more rare use case.

### Combining thread and SIMD loop annotations
Ripple supports the combination of multi-threading and SIMD loop annotations.
They can of course be used on different loops in the same code,
but they can also be used on the same loop.
The constraint is that the thread Ripple annotations needs to be called
_before_ the Ripple SIMD annotations,
as in the following multi-threaded vector addition example:
```C
#include <ripple.h>
#include <ripple/thread.h>
#define VECTOR_PE 0
void * vecadd(void * vargs) {
  args_t * args = (args_t *)vargs;
  size_t n = args->n;
  float * a = args->a;
  float * b = args->b;
  float * sum = args->sum;
  // By using `RIPPLE_THD_DYNAMIC`, we use all the threads available (i.e. 16).
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*block*/0, /*n_dims*/RIPPLE_THD_DYNAMIC);
  ripple_block_t simdb = ripple_set_block_shape(VECTOR_PE, 32);

  ripple_thd_parallel(thdb, /*chunk_size*/32, /*flags*/RIPPLE_THD_PAR_DFT, /*dims*/0);
  ripple_parallel(simdb, /*dims*/0);
  for (size_t i = 0; i < n; ++i) {
      sum[i] = a[i] + b[i];
  }
}
```

## Barrier synchronization


### Most common use case

The most basic use case for a barrier is to synchronize subsequent parallel loops.
In the following example, the `i` loop produces some values in `tmp`,
which are consumed by statements in a following `k` loop.

Since we don't know which thread will access which elements of `tmp`
(because we don't know `f()`), we need to wait for the `i` loop to be
finished before we start the `k` loop.
We make this wait happens using `ripple_thd_barrier()`.

```C
runtime_t *rt; // assume it's defined elsewhere
void typical_barrier(size_t n, float * A, float * B) {
  float tmp[n];
  ripple_thd_block_t thdb =
    ripple_thd_set_block_shape(rt, 0, 1, RIPPLE_THD_DYNAMIC);
  ripple_thd_parallel(thdb, 32, RIPPLE_THD_PAR_DFT, 0);
  for (size_t i = 0; i < n; ++i) {
    tmp[f(i)] = sqrtf(B[i]);
  }
  // Make all threads along dimension 0 (the only dimension) join a barrier
  ripple_thd_barrier(thdb, /*dimensions*/0b1);

  ripple_thd_parallel(thdb, 32, RIPPLE_THD_PAR_DFT, 0);
  for (size_t k = 0; k < n; ++k) {
    B[i] = tmp[i] * tmp[n - i - 1];
  }
}
```
### Multi-dimensional loops

Consider the following example, where the computations of a double-nested loop.
Imagine that we want to distribute its computations across threads using Ripple.

```C
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j <n; ++j) {
      A[f(i), j] += A[i, j];
    }
  }
```

We can see that for a given iteration `i`, all iterations of `j` are independent.
However, since `f(i)` is unknown, we must assume that consecutive iterations of
loop `i` carry a dependence (hence we cannot parallelize it).

Hence we parallelize loop `j` along a one-dimensional thread block,
using `ripple_thd_parallel`.
But since consecutive values of `i` are not independent, we need to force
all the threads distributed along `j` to wait each time the `j` loop has run.
To enforce this, we would add a `ripple_thd_barrier()` call after loop `j`,
as follows:

```C
  for (size_t i = 0; i < n; ++i) {
    ripple_thd_parallel(thdb, 32, RIPPLE_THD_DYNAMIC);
    for (size_t j = 0; j <n; ++j) {
      A[f(i), j] += A[i, j];
    }
    ripple_thd_barrier(thdb, 0b1);
  }
```

### Multi-dimensional thread blocks

The two previous examples had only one thread dimension, hence the only
possible value for `ripple_thd_barrier`'s `dims` parameter is "dimension 0",
i.e. `0b1`.

The `ripple_thd` API offers multi-dimensional threading,
i.e. multi-threading over a multi-dimensional block of threads.

Multi-dimensional threading can be useful in many cases, including:
1. when we want to exploit the parallelism of several nested loops;
2. when we want to represent a heterogeneous multi-threading architecture
3. when we only want subsets of the threads to synchronize with each other.

To make things concrete, let us look at an example where a barrier
synchronization is needed in a double-nested parallel loop nest.

Let's assume that we want to reduce the values of A along its second dimension,
and that you don't want to implement the reduction using atomics (at all).

Function `sum_along_dim1` below performs this by distributing
the iterations of `i` along dimension 0 of the Ripple thread block,
and also distributing the iterations of `j` along dimension 1
of that same thread block.

For each value of `i`, each thread [*,t1] accumulates `chunk_size`
values of A[i][*] sequentially into `scratch[i][t1]`.
When loop `j` finishes, accumulations are all done along `j` into `nt1`
partial sums (in `scratch[i]`).
To finish the reduction along `j`, we need to sum the partial sums from
`scratch[i]` into a scalar, which we store in `B[i]`.

Since we don't want to use atomics for the summation in this example,
we wait for all the partial sums to be computed, using ripple_thd_barrier.
Notice that the "dimensions" parameter here is `along_t1`, i.e. 0b10.
This means that a barrier along dimension 1 is executed,
i.e. all threads with the same value for `t0` (and different values for `t1`)
wait for each other.
Finally, we let thread for which `t1 = 0` (the "main thread" along `j`) perform
the summation.

```C++
/// @brief 1-d reduction in a 2-d thread environment,
/// using a atomic-less reduction along dim 1
static void *sum_along_dim1(int (*A)[N], int *B, int *scratch[8],
                         unsigned chunk_size) {
  constexpr size_t nt1 = 8;
  constexpr unsigned along_t1 = 0b10;
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(args->rt, /*block*/ 0,
                                                       /*n_dims*/ 2,
                                                       RIPPLE_THD_DYNAMIC, nt1);
  size_t t0 = ripple_thd_id(thdb, 0);
  size_t t1 = ripple_thd_id(thdb, 1);
  ripple_thd_parallel(thdb, chunk_size, RIPPLE_THD_FLAG_DFT, 0);
  for (unsigned i = 0; i < N; ++i) {
    scratch[i][t1] = 0;
    // First, do a data-parallel accumulation along j for each thread.
    ripple_thd_parallel(thdb, chunk_size, RIPPLE_THD_PAR_DFT, 1);
    for (unsigned j = 0; j < i; ++j) {
      scratch[i][t1] += A[i][j];
    }
    ripple_thd_barrier(thdb, along_t1); // Barrier along dimension 1
    // Now for the sequential glue code, done by threads (t0, 0)
    if (ripple_thd_is_main(thdb, along_t1)) {
      B[i] = 0;
      for (unsigned j = 0; j < nt1; ++j) {
        B[i] += scratch[i][j];
      }
    }
  }

  return nullptr;
}

```
An interesting aspect of this example is that it mixes loop annotations
(`ripple_thd_parallel()`) with direct use of SPMD indices (`t0`, `t1`).

The `sum_along_dim1` example demonstrates the use of a 1-dimensional barrier
inside a 2-dimensional thread block.
Ripple supports barriers that involve arbitrary subsets of the block dimensions.
`-1` can be used to indicate that the barrier should involve all dimensions,
regardless of the number of dimensions in the thread block.

Ripple supports up to 3 thread dimensions.

## Important differences between SIMD and thread Ripple
The threading API is distinguished from the SIMD API by its prefix:
`ripple_thd_`, as opposed to SIMD's mere `ripple_` prefix.

### Static SIMD, Dynamic threading
The following table summarizes some of the major differences between Ripple thread blocks for SIMD and for threads.
Following sections provide additional detail.

| Ripple API | What it is | initialization cost | when to initialize |
|------------|------------|---------------------|--------------------|
| `ripple_block_t` | compiler abstraction to convey SIMD properties| None| At least once per function (can sometimes be passed to inlined functions and some vector lib functions)|
| `ripple_thd_block_t` | runtime object to convey threading properties | Synchronization cost | When you need to modify the thread block shape. Otherwise, pass it around functions.|

### Single active thread block
SIMD targets can require the use of several SIMD engines
(matrix, vector, scalar) in a single function.
To enable this, Ripple supports the use of several Ripple blocks
within a function. This is detailed in sections about SIMD vectorization.

In contrast, threading environments are better represented as a uniform block,
for two main reasons:
- The cost of spawning threads is typically non-trivial.
Hence most efficient multi-threaded runtimes start threads
and express multi-threading programs within an environment
where threads are already running.

- The underlying hardware threads are not necessarily uniform
or homogeneous, they can be hierarchical,
and support different execution models
(e.g. separate control processor + grid of compute processors,
vs. homogeneous processor grid).
All these configurations can be represented using a homogeneous block hierarchy.

Ripple implements this distinction for execution units with a costly transition
between parallel and sequential executions, such as multi-thread and multi-core
systems.
Hence, for these levels of parallelism, all blocks declared with `ripple_thd_set_block_shape()` are a representation of the same threads from the
underlying runtime.
The block's shape can be changed by declaring a new thread block,
but all blocks associated with a given underlying runtime object
use the same number of underlying threads.

### A thread block is a dynamic data structure
While Ripple needs to perform a compiler transformation to turn SPMD code
representing SIMD computations into actually SIMD code,
thread parallelism doesn't require any compiler transformation.
Instead, it is almost entirely managed by runtime code.

This is because thread parallelism is typically available through a threading
runtime (e.g. POSIX threads, QuRT, QHPI).


## Related work
### Abstracted thread management
There are many multi-threading programming paradigms out there.
Ripple is closest to OpenMP(R) as it offers a loop annotation
that specifies thread parallelism and scheduling.
A broader (and more complex) palette of static and
dynamic scheduling is available in OpenMP.

SPMD is also available in OpenMP, within the scope of _parallel regions._

The Intel(R) ISPC compiler offers the SPMD programming model for multi-threading.

### Load balancing
When the per-iteration load can be calculated statically by a compiler,
a load-balanced partitioning of iterations can be determined
by creating variable-sized chunks.
This optimization technique, called Algebraic Tiling __[1]__,
requires complex compiler calculations and is not supported by Ripple.

The oneAPI Thread Building Blocks (R) also offer multi-threading abstractions,
and a work-stealing-based scheduling algorithm.


---
__[1]__ C. Rosetti, Ph. Clauss, "Algebraic Tiling," IMPACT 2023,
13th International Workshop on Polyhedral Compilation Techniques.

OpenMP is a registered trademark of the OpenMP Architecture Review Board.

The Intel (R) ISPC Compiler is a trademark of Intel Corporation.

The oneAPI Thread Building Blocks (R) is a trademark of Intel Corporation.
