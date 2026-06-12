# Optimizing Ripple programs

While the Specification section presents the syntax and API of Ripple,
this section focuses on how to achieve good performance using Ripple.
We start by going over the [requirements for an efficient utilization of the
underlying vector hardware](./vector-principles.md).
Then, we present a few aspects related to
[expressing coalesced accesses](./coalescing.md), which require special attention.
Finally, we take a look at ways to
[leverage some of the strengths of Hexagon(R)'s HVX instruction set](https://qualcomm.github.io/learn-ripple/opt/hexagon/hvx-opt.html)
through Ripple.

---
Hexagon is a registered trademark of Qualcomm Incorporated.

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
