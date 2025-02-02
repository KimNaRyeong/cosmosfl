FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

WORKDIR /

RUN apt update && apt upgrade -y && apt install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0

CMD ["ollama", "serve"]