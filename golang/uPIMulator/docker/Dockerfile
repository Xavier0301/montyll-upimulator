FROM --platform=linux/amd64 upmem_sdk_2023:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git
RUN apt install -y wget
RUN apt install -y cmake
RUN apt install -y ninja-build
RUN apt install -y pkg-config
RUN apt install -y python3
RUN apt install -y python3-pip

# UPMEM SDK
WORKDIR /root
RUN echo "source /upmem-sdk-2023.4.0/upmem_env.sh" >> /root/.bashrc

RUN cp -r /upmem-sdk-2023.4.0 /root/ && mv upmem-sdk-2023.4.0 upmem-2023.2.0-Linux-x86_64


WORKDIR /root/uPIMulator
