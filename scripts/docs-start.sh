#!/bin/bash

# Format the generated HTML to make it more readable
# npm install -g js-beautify
# js-beautify --type html docs/index.html > docs/index-formatted.html

# Start web server locally to check generated docs
cd docs
python -m http.server 8000