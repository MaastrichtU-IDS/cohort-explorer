#!/bin/bash

# pip install --upgrade pip

if [ ! -d ".venv" ]; then
    echo ".venv virtual environment does not exist. Creating it"
    python -m venv .venv
fi

echo "Activating virtual environment"
source .venv/bin/activate

# http://lambdamusic.github.io/Ontospy
pip install ontospy

# https://github.com/dgarijo/Widoco
wget -O widoco.jar https://github.com/dgarijo/Widoco/releases/download/v1.4.20/widoco-1.4.20-jar-with-dependencies_JDK-17.jar

# https://github.com/stain/owl2jsonld
wget -O owl2jsonld.jar https://github.com/stain/owl2jsonld/releases/download/0.2.1/owl2jsonld-0.2.1-standalone.jar