# =============================================================================
# Stage 1: Builder — download SDK, clone LLVM, build Hexagon toolchain
# =============================================================================
FROM ubuntu:24.04 AS sources

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    unzip \
    cmake \
    ninja-build \
    python3 \
    clang \
    lld \
    make \
    python-is-python3 \
    libc++-dev \
    libc++abi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Hexagon SDK
# ---------------------------------------------------------------------------
WORKDIR /build

ENV HEXAGON_SDK_ROOT_DL=/build/Hexagon_SDK/6.5.0.0

RUN wget -q "https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip" \
    -O Hexagon_SDK_Linux.zip \
    && unzip -q Hexagon_SDK_Linux.zip \
    && rm Hexagon_SDK_Linux.zip \
    && cd ${HEXAGON_SDK_ROOT_DL} \
    && ./utils/scripts/SDKDependencyDownloader --sdk_root ${HEXAGON_SDK_ROOT_DL} \
    # Strip the SDK down to what's actually used:
    #   - Tools/target/hexagon  (sysroot for compiler-rt and test-suite builds)
    #   - Tools/bin/hexagon-sim (cross-compile emulator)
    #   - Tools/lib/libedit.so* + iss/ + libclade*.so  (hexagon-sim runtime deps)
    #   - setup_sdk_env.source + utils/scripts/  (env setup)
    #   - ipc/fastrpc/qaic/  (compiled by setup_sdk_env.source)
    && TOOLS=${HEXAGON_SDK_ROOT_DL}/tools/HEXAGON_Tools/19.0.07/Tools \
    && rm -rf \
        ${HEXAGON_SDK_ROOT_DL}/docs \
        ${HEXAGON_SDK_ROOT_DL}/libs \
        ${HEXAGON_SDK_ROOT_DL}/incs \
        ${HEXAGON_SDK_ROOT_DL}/examples \
        ${HEXAGON_SDK_ROOT_DL}/tools/android-ndk-r25c \
        ${HEXAGON_SDK_ROOT_DL}/tools/HEXAGON_Tools/19.0.07/Uninstall \
        ${HEXAGON_SDK_ROOT_DL}/tools/HEXAGON_Tools/19.0.07/Documents \
        ${HEXAGON_SDK_ROOT_DL}/tools/HEXAGON_Tools/19.0.07/Examples \
        ${HEXAGON_SDK_ROOT_DL}/tools/HALIDE_Tools \
        ${HEXAGON_SDK_ROOT_DL}/tools/debug \
        ${HEXAGON_SDK_ROOT_DL}/tools/elfsigner \
        ${HEXAGON_SDK_ROOT_DL}/tools/qhcg \
        ${HEXAGON_SDK_ROOT_DL}/tools/Tools/QEMUHexagon \
        ${HEXAGON_SDK_ROOT_DL}/utils/visualize_hvx_instructions \
    && find ${TOOLS}/bin -mindepth 1 -maxdepth 1 \( -type f -o -type l \) ! -name 'hexagon-sim' -delete \
    && find ${TOOLS}/lib -mindepth 1 -maxdepth 1 ! -name 'libedit*' ! -name 'iss' ! -name 'libclade*' -exec rm -rf {} +

# ---------------------------------------------------------------------------
# LLVM source
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/qualcomm/Ripple.git /build/llvm-project
RUN git clone --depth 1 https://github.com/qualcomm/eld.git /build/eld

FROM sources AS build-llvm

# ---------------------------------------------------------------------------
# Build Hexagon clang (compiler only, no runtime yet)
# ---------------------------------------------------------------------------
RUN cmake -G Ninja \
    -S /build/llvm-project/llvm \
    -B /build/build-hexagon \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    "-DLLVM_ENABLE_PROJECTS=clang" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_TARGETS_TO_BUILD="Hexagon" \
    -DLLVM_DEFAULT_TARGET_TRIPLE=hexagon-unknown-unknown-elf \
    -DCMAKE_INSTALL_PREFIX=/opt/hexagon \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DCLANG_LINK_CLANG_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DCMAKE_INSTALL_RPATH=/opt/hexagon/lib \
    && cmake --build /build/build-hexagon \
    && cmake --build /build/build-hexagon -t install \
    && rm -rf /build/build-hexagon

FROM build-llvm AS build-eld

# Flip LLVM_LINK_LLVM_DYLIB so ELD uses static component .a files instead of libLLVM.so
RUN sed -i 's/set(LLVM_LINK_LLVM_DYLIB ON)/set(LLVM_LINK_LLVM_DYLIB OFF)/' \
    /opt/hexagon/lib/cmake/llvm/LLVMConfig.cmake

# ---------------------------------------------------------------------------
# Build ELD statically against the installed clang/LLVM
# ---------------------------------------------------------------------------
RUN cmake -G Ninja \
    -S /build/eld \
    -B /build/build-eld \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=/opt/hexagon/lib/cmake/llvm \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DCLANG_LINK_CLANG_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DELD_USE_EXTERNAL_LLVM=ON \
    -DELD_TARGETS_TO_BUILD="Hexagon" \
    -DLLVM_TABLEGEN_EXE=/opt/hexagon/bin/llvm-tblgen \
    -DCMAKE_INSTALL_PREFIX=/opt/hexagon \
    && cmake --build /build/build-eld \
    && cmake --install /build/build-eld \
    && rm -rf /build/build-eld

# Restore LLVM_LINK_LLVM_DYLIB so compiler-rt and downstream stages link against libLLVM.so as intended
RUN sed -i 's/set(LLVM_LINK_LLVM_DYLIB OFF)/set(LLVM_LINK_LLVM_DYLIB ON)/' \
    /opt/hexagon/lib/cmake/llvm/LLVMConfig.cmake

FROM build-eld AS build-compiler-rt

SHELL ["/usr/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Build Hexagon compiler-rt for Hexagon
# ---------------------------------------------------------------------------
RUN . ${HEXAGON_SDK_ROOT_DL}/setup_sdk_env.source && \
    cmake -G Ninja \
        -S /build/llvm-project/compiler-rt \
        -B /build/build-hexagon-rt \
        -DLLVM_DIR=/opt/hexagon/lib/cmake/llvm \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=/opt/hexagon/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/hexagon/bin/clang++ \
        -DCOMPILER_RT_OS_DIR= \
        -DCOMPILER_RT_BUILD_BUILTINS=OFF \
        -DCOMPILER_RT_BUILD_RIPPLE=ON \
        -DCOMPILER_RT_BUILD_CRT=OFF \
        -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
        -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
        -DCOMPILER_RT_BUILD_MEMPROF=OFF \
        -DCOMPILER_RT_BUILD_ORC=OFF \
        -DCOMPILER_RT_BUILD_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
        -DCOMPILER_RT_BUILD_XRAY=OFF \
        -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
        -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
        -DCMAKE_C_COMPILER_TARGET=hexagon-unknown-unknown-elf \
        -DCMAKE_CXX_COMPILER_TARGET=hexagon-unknown-unknown-elf \
        "-DCMAKE_CXX_FLAGS=-stdlib=libc++ -D_USE_GENERIC_XLOCALE_SHIMS -D_LIBCPP_PROVIDES_DEFAULT_RUNE_TABLE -D_C99 -D_C_AS_CPP -D_HAS_GENERIC_TEMPLATES=0" \
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=eld" \
        -DCMAKE_SYSROOT=${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/target/hexagon \
    && cmake --build /build/build-hexagon-rt \
    && cmake --install /build/build-hexagon-rt --prefix $(/opt/hexagon/bin/clang --print-resource-dir) \
    && rm -rf /build/build-hexagon-rt

# Strip headers, cmake integration files, and static libs — not needed at runtime
RUN rm -rf /opt/hexagon/include \
           /opt/hexagon/lib/cmake \
           /opt/hexagon/lib/*.a

# =============================================================================
# Stage 2: Runtime — minimal Ubuntu 24 image with just the toolchain
# =============================================================================
FROM ubuntu:24.04 AS toolchain

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    cmake \
    ninja-build \
    python3 \
    make \
    python-is-python3 \
    libc++-dev \
    libc++abi-dev \
    bash \
    wget \
    vim \
    powerline fonts-powerline \
    bash-completion \
    fzf \
    bat \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install libncurses.so.5 and libtinfo.so.5 from Ubuntu 22.04 (Jammy)
RUN wget -q http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libncurses5_6.3-2ubuntu0.1_amd64.deb \
    && wget -q http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && apt-get install -y --no-install-recommends ./libncurses5_6.3-2ubuntu0.1_amd64.deb ./libtinfo5_6.3-2ubuntu0.1_amd64.deb \
    && rm -rf libncurses5_6.3-2ubuntu0.1_amd64.deb libtinfo5_6.3-2ubuntu0.1_amd64.deb

# Copy the installed Hexagon toolchain
COPY --link --from=build-compiler-rt /opt/hexagon /opt/hexagon

# Copy the Hexagon SDK (needed at test-suite build time for headers/libs)
COPY --link --from=build-compiler-rt /build/Hexagon_SDK/6.5.0.0 /opt/hexagon-sdk

# Create a non-root user
RUN useradd -m -s /bin/bash ripple

# Download mdbook for building and serving documentation
ARG MDBOOK_VERSION=v0.4.52
RUN wget -q "https://github.com/rust-lang/mdBook/releases/download/${MDBOOK_VERSION}/mdbook-${MDBOOK_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
    -O /tmp/mdbook.tar.gz \
    && tar -xzf /tmp/mdbook.tar.gz -C /usr/local/bin \
    && rm /tmp/mdbook.tar.gz

USER ripple
WORKDIR /home/ripple

FROM toolchain AS test-suite

# ---------------------------------------------------------------------------
# Clone the test suite and build it for Hexagon (standalone / sim)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/qualcomm/ripple-test-suite.git /home/ripple/ripple-test-suite \
    && rm -rf /home/ripple/ripple-test-suite/.git

SHELL ["/usr/bin/bash", "-c"]

ARG HEXAGON_ARCH=81

RUN . /opt/hexagon-sdk/setup_sdk_env.source && \
    cmake -G Ninja \
        -S /home/ripple/ripple-test-suite \
        -B /home/ripple/ripple-test-suite/build \
        -DCMAKE_SYSTEM_NAME=HexagonStandalone \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/hexagon/bin/clang++ \
        "-DCMAKE_CXX_FLAGS=--target=hexagon-unknown-unknown-elf -O2 -mv${HEXAGON_ARCH} -mhvx -mhvx-length=128B -std=c++11 -stdlib=libstdc++" \
        -DCMAKE_SYSROOT=${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/target/hexagon \
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=eld" \
        "-DCMAKE_CROSSCOMPILING_EMULATOR=${DEFAULT_HEXAGON_TOOLS_ROOT}/Tools/bin/hexagon-sim;--simulated_returnval;--" \
    && cmake --build /home/ripple/ripple-test-suite/build

# Clone and build documentation (three books)
RUN git clone --depth 1 https://github.com/qualcomm/learn-ripple.git /home/ripple/learn-ripple \
    && rm -rf /home/ripple/learn-ripple/.git
RUN mdbook build -d build /home/ripple/learn-ripple \
    && mdbook build -d build /home/ripple/learn-ripple/opt/hexagon \
    && mdbook build -d build /home/ripple/learn-ripple/troubleshooting

# Shell setup — kept after the build so config changes don't bust the build cache
COPY --chown=ripple:ripple --chmod=644 bashrc /tmp/bashrc.extra
COPY --chown=ripple:ripple --chmod=644 welcome /home/ripple/.welcome
COPY --chown=ripple:ripple --chmod=600 bash_history /home/ripple/.bash_history
RUN cat /tmp/bashrc.extra >> /home/ripple/.bashrc && rm /tmp/bashrc.extra

CMD ["/bin/bash", "--login", "-i"]
