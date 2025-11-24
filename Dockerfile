# syntax=docker/dockerfile:1.6

FROM python:3.12-slim

ENV UV_LINK_MODE=copy
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.91.1

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN mkdir -p /root/.cache/uv /app/.venv

VOLUME /app/.venv
VOLUME /root/.cache

# Build tools for C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install rust toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Install Node.js 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g npm@11.6.2 \
    && npx -y promptfoo@latest --help

# Copy only dependency metadata first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into .venv (but not the project code yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Now copy the rest of the project
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
