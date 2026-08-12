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

# Download a GitHub release asset and verify we actually got a jar (zip
# archive) rather than an HTML error/rate-limit page — a corrupt jar otherwise
# only surfaces later, in the middle of the docs build.
#
# Anonymous downloads from the release CDN regularly return 503 to CI runners,
# so when GITHUB_TOKEN is set (always the case in GitHub Actions) the asset is
# fetched through the authenticated API instead, which is far more reliable.
download_jar() {
    local repo="$1"
    local tag="$2"
    local asset_name="$3"
    local out="$4"
    local auth=()
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    fi

    # Resolve the asset id via the API, then download the asset itself.
    local asset_id
    asset_id=$(curl -fsSL --retry 3 --retry-delay 5 "${auth[@]}" \
        "https://api.github.com/repos/$repo/releases/tags/$tag" \
        | python -c "import json,sys; r=json.load(sys.stdin); print(next(a['id'] for a in r['assets'] if a['name'] == '$asset_name'))")
    curl -fL --retry 5 --retry-delay 10 "${auth[@]}" \
        -H "Accept: application/octet-stream" \
        -o "$out" "https://api.github.com/repos/$repo/releases/assets/$asset_id"

    if ! unzip -t "$out" > /dev/null 2>&1; then
        echo "ERROR: $out is not a valid jar (download of $asset_name from $repo@$tag failed or was corrupted)" >&2
        exit 1
    fi
}

# https://github.com/dgarijo/Widoco
download_jar dgarijo/Widoco v1.4.20 widoco-1.4.20-jar-with-dependencies_JDK-17.jar widoco.jar

# https://github.com/stain/owl2jsonld
download_jar stain/owl2jsonld 0.2.1 owl2jsonld-0.2.1-standalone.jar owl2jsonld.jar