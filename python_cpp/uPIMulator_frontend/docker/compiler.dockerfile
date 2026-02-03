FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/root/upmem_linker/src:$PYTHONPATH"

RUN chmod 1777 /tmp

RUN apt update
RUN apt install -y git
RUN apt install -y wget
RUN apt install -y vim
RUN apt install -y tmux
RUN apt install -y mlocate
RUN apt install -y cmake
RUN apt install -y ninja-build
RUN apt install -y pkg-config
RUN apt install -y libnuma-dev
RUN apt install -y libelf-dev
RUN apt install -y flex

# Python 3.10
# RUN apt update
# RUN apt upgrade -y
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt update
# RUN apt install -y python3.10
# RUN apt install -y python3-pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 && \
    apt-get install -y curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# UPMEM LLVM
WORKDIR /root
RUN git clone https://github.com/upmem/llvm-project.git
RUN mkdir -p /root/llvm-project/build
WORKDIR /root/llvm-project/build
RUN cmake -G Ninja /root/llvm-project/llvm -DLLVM_ENABLE_PROJECTS="clang"
RUN cmake build .

# UPMEM SDK
WORKDIR /root
RUN wget http://sdk-releases.upmem.com/2025.1.0/ubuntu_20.04/upmem-2025.1.0-Linux-x86_64.tar.gz
RUN tar -zxvf upmem-2025.1.0-Linux-x86_64.tar.gz
RUN echo "source /root/upmem-2025.1.0-Linux-x86_64/upmem_env.sh" > /root/.bashrc

WORKDIR /root/upmem_compiler
