#!/bin/bash
set -e

# pip install --upgrade pip

if [ ! -d ".venv" ]; then
    echo ".venv virtual environment does not exist. Creating it"
    python -m venv .venv
fi

echo "Activating virtual environment"
source .venv/bin/activate

# http://lambdamusic.github.io/Ontospy
pip install ontospy

# Download a release jar with retries, and verify we actually got a jar (zip
# archive) rather than an HTML error/rate-limit page — a corrupt jar otherwise
# only surfaces later, in the middle of the docs build.
download_jar() {
    local url="$1"
    local out="$2"
    curl -fL --retry 5 --retry-delay 10 -o "$out" "$url"
    if ! unzip -t "$out" > /dev/null 2>&1; then
        echo "ERROR: $out is not a valid jar (download from $url failed or was corrupted)" >&2
        exit 1
    fi
}

# https://github.com/dgarijo/Widoco
download_jar "https://github.com/dgarijo/Widoco/releases/download/v1.4.20/widoco-1.4.20-jar-with-dependencies_JDK-17.jar" widoco.jar

# https://github.com/stain/owl2jsonld
download_jar "https://github.com/stain/owl2jsonld/releases/download/0.2.1/owl2jsonld-0.2.1-standalone.jar" owl2jsonld.jar