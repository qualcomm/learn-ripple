# Multi-threading in Ripple
Ripple allows you to use the SPMD and loop annotation parallel
programming models to express multi-threaded
(sometimes also called "self-threaded") programs.

## Important differences between SIMD and thread Ripple
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
Hence, for these levels of parallelism, instead of arbitrary numbers of blocks,
Ripple implements an "ambient" block, much like in OpenMP(R)'s parallel regions
or in CUDA and OpenCL.
The block's shape can be changed, but for every underlying parallel runtime,
all code assumes the same block shape.

### A thread block is a dynamic data structure
While Ripple needs to perform a compiler transformation to turn SPMD code
representing SIMD computations into actually SIMD code,
thread parallelism doesn't require any compiler transformation.
Instead, it is almost entirely managed by runtime code.

This is because thread parallelism is typically available through a threading
runtime (e.g. POSIX threads, QuRT, QHPI).

| Ripple API | What it is | initialization cost | when to initialize |
|------------|---|---|---|
| `ripple_block_t` | compiler abstraction to convey SIMD properties| None| At least once per function (can sometimes be passed to inlined functions and some vector lib functions)|
| `ripple_thd_block_t` | runtime object to convey threading properties | Synchronization cost | When you need to modify the thread block shape. Otherwise, pass it around functions.|

## Ripple threading API
The Ripple thread API is prefixed with `ripple_thd`.
Just like for SIMD, you can declare a block (of threads),
retrieve the size of the block along any dimension,
and get the index of your current thread along any dimension.
You can also annotate loops to distribute their iterations along one or more
loops using the `ripple_thd_parallel` API,
which supports static and dynamic scheduling.

As always, the loop annotation style is syntactic sugar provided by Ripple
on top of SPMD. As a result, both styles can be mixed if necessary.

### SPMD API
```C
#include <ripple/thread.h>

/// \brief Defines a thread block shape for the threads provided through
/// the \p underlying_runtime threading runtime object.
/// \param n_dims the number of dimensions of the thread block shape
/// \param shape the thread block shape.
/// A special constant, `RIPPLE_THD_DYNAMIC` can be used in one of the dimensions
/// to mean "as many as the number of underlying threads allow".
/// Currently, for `RIPPLE_THD_DYNAMIC` to work,
/// there must exist a value for which the volume of the resulting
/// shape matches the number of spawned threads exactly
/// (which is always true for one-dimensional shapes)
ripple_thd_block_t ripple_thd_set_block_shape(void *underlying_runtime,
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

For each supported threading mechanism, Ripple uses a different type of
underlying threading runtime object.
These can also be useful to save boilerplate thread initialization and
termination code.
Supported runtimes and their underlying runtime are represented
in the following table:

| Runtime | Underlying Type | Include `ripple/thread.h` and compile with |
|---------|----------------|-----------|
| QuRT    | `qthd_runtime_t` | `-DRIPPLE_THD_USE_QURT` |
| QHPI    | `QHPI_RuntimeHandle` | `-DRIPPLE_THD_USE_QHPI` |

How underlying runtimes are initialized, and how the threads are spawned,
are defined on a per-runtime basis.
For instance, QHPI itself offers an environment in which the threads are
already running, and the underlying runtime object is already initialized.
We will see how to create an underlying runtime object and spawn threads
for QuRT below.

### Loop annotations API
```C
#include <ripple/thread.h>

/// \brief Tells Ripple to statically distribute the iterations of the for loop
/// immediately following this call.
/// Chunks of \p chunk_size contiguous iterations get distributed cyclically
/// across dimension \p dims of the thread block.
/// \param no_wait Future parameter that will toggle an implicit barrier
///        at the end of the loop.
///        Currently assumed to be 1 (no barrier gets executed).
///        To ensure forward compatibility, please set it to 1 (no wait).
///
/// A static distribution determines a fixed mapping from iteration chunks
/// to threads. It has virtually no runtime overhead and its per-thread
/// execution is deterministic. However, it can suffer from load imbalance
/// when the amount of work in the chunks is too heterogeneous.
void ripple_thd_parallel(ripple_thd_block_t b, unsigned chunk_size, int no_wait, unsigned ... dims);

/// \brief Tells Ripple to dynamically distribute the iterations of the for loop
/// immediately following this call.
/// Chunks of \p chunk_size iterations get distributed dynamically across
/// dimension \p dims of the thread block.
/// Only dimension 0 is supported.
//  This is enough to express all degrees of dynamic parallelism necessary.
/// \param no_wait Future parameter that will toggle an implicit barrier
///        at the end of the loop.
///        Currently assumed to be 1 (no barrier gets executed).
///        To ensure forward compatibility, please set it to 1 (no wait).
///
/// A dynamic distribution lets threads process a chunk
/// whenever they are done with the previous one, making the execution tolerant
/// to heterogeneous chunks (i.e, more load-balanced).
/// The downside of dynamic distributions is that
/// they come with some runtime overhead.
void ripple_thd_parallel_dyn(ripple_thd_block_t b, unsigned chunk_size, int no_wait, unsigned ... dims);
```

### QuRT-specific API
While the API above is universal across runtimes, we rely on a few runtime-specific APIs to make the use of multi-threaded Ripple easier for
some runtimes.

```C
/// \brief Initializes the Ripple threading runtime from an underlying runtime (for QuRT, a `qthd_runtime_t *`).
ripple_thd_block_t ripple_thd_init(int PE_id, void *underlying);

/// Destroys the block formed from this QuRT underlying runtime.
void ripple_thd_exit(ripple_thd_block_t block);

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

## SPMD programming model for threads
To create a multi-threaded function with the SPMD model,
we need a thread block object.
We can then access the SPMD thread indices and barriers from that block.

In Ripple threads, the thread block is a runtime object,
which can be passed around among functions.

The following example code implements a simple multi-threaded,
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
#define NUM_THREADS 16
#define QURT_THREADS 0
#define N 1024
qthd_runtime_t *rt;

float A[N], B[N], SUM[N];

typedef struct {
  size_t n;
  float * a;
  float * b;
  float * sum;
} args_t;

void * vecadd(void * vargs) {
  args_t * args = (args_t *)vargs;
  size_t n = args->n;
  float * a = args->a;
  float * b = args->b;
  float * sum = args->sum;
  // By using `RIPPLE_THD_DYNAMIC`, we use all the threads available (i.e. 16).
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, 1, RIPPLE_THD_DYNAMIC);
  size_t thd_id = ripple_thd_id(thdb, /*dimension*/ 0);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  for (size_t i = thd_id * chunk_size; i < (thd_id + 1) * chunk_size; ++i) {
    if (i < n)
      sum[i] = a[i] + b[i];
  }
}

int main() {
  rt = qthd_runtime_init(NUM_THREADS);
  ripple_thd_block_t b = ripple_thd_init(QURT_THREADS, rt);
  args_t args = {N, A, B, SUM};
  ripple_thd_call(b, vecadd, &args); // synchronous call, args doesn't escape
  qthd_runtime_exit(rt);
}
```

QHPI offers an ambient multi-threaded environment and passes its runtime
object directly to the kernels.
The same SPMD vector addition kernel would look like below.

```C++
#include "HTP/core/qhpi.h"
#include <ripple/thread.h>

uint32_t vecadd(QHPI_RuntimeHandle *rt, uint32_t n_outputs, QHPI_Tensor ** outputs, uint32_t n_inputs, const QHPI_Tensor *const * inputs) {
  // _________ Begin QHPI boilerplate parameter unpacking __________
  const float * a = (const float *) qhpi_tensor_raw_data(inputs[0]);
  const float * b = (const float *) qhpi_tensor_raw_data(inputs[1]);
  float * sum = (float *) qhpi_tensor_raw_data(outputs[0]);
  QHPI_Shape shape = qhpi_tensor_shape(outputs[0]);
  uint32_t n = shape_size(shape);
  // _______________ End QHPI parameter unpacking __________________

  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*n_dims*/1, RIPPLE_THD_DYNAMIC);
  size_t thd_id = ripple_thd_id(thdb, /*dimension*/ 0);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  for (size_t i = thd_id * chunk_size; i < (thd_id + 1) * chunk_size; ++i) {
    if (i < n)
      sum[i] = a[i] + b[i];
  }
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
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*n_dims*/1, /*shape*/RIPPLE_THD_DYNAMIC);
  size_t n_thd = ripple_thd_get_block_size(thdb, /*dimension*/0);
  size_t chunk_size = (n + n_thd - 1) / n_thd;
  ripple_thd_parallel(thdb, chunk_size, /*nowait*/1, /*dims*/0);
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
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*n_dims*/1, RIPPLE_THD_DYNAMIC);
  ripple_thd_parallel_dyn(thdb, /*chunk_size*/16, /*nowait*/1, /*dims*/0);
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
  ripple_thd_block_t thdb = ripple_thd_set_block_shape(rt, /*n_dims*/RIPPLE_THD_DYNAMIC);
  ripple_block_t simdb = ripple_set_block_shape(VECTOR_PE, 32);

  ripple_thd_parallel(thdb, /*chunk_size*/32, /*nowait*/1, /*dims*/0);
  ripple_parallel(simdb, /*dims*/0);
  for (size_t i = 0; i < n; ++i) {
      sum[i] = a[i] + b[i];
  }
}
```

# Related work
## Abstracted thread management
There are many multi-threading programming paradigms out there.
Ripple is closest to OpenMP(R) as it offers a loop annotation
that specifies thread parallelism and scheduling.
A broader (and more complex) palette of static and
dynamic scheduling is available in OpenMP.

SPMD is also available in OpenMP, within the scope of _parallel regions._

The ISPC compiler offers the SPMD programming model for multi-threading.

## Load balancing
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

