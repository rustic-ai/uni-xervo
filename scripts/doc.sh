#!/usr/bin/env bash
set -euo pipefail

# Build both MkDocs website and rustdoc API reference, then merge them
# into a single site under website/site/.
#
# After running this script, open website/site/index.html for the docs
# or website/site/api/uni_xervo/index.html for the API reference.

echo "Building rustdoc..."
# Use explicit feature list instead of --all-features because
# provider-mistralrs and gpu-cuda require a CUDA toolkit at build time.
DOC_FEATURES="provider-candle,provider-fastembed,provider-openai,provider-gemini,provider-vertexai,provider-mistral,provider-anthropic,provider-voyageai,provider-cohere,provider-azure-openai"
cargo doc --no-deps --features "$DOC_FEATURES"

echo "Copying rustdoc into website/docs/api/..."
rm -rf website/docs/api
cp -r target/doc website/docs/api

echo "Building MkDocs site..."
(cd website && poetry run mkdocs build --strict)

echo "Done. Open website/site/index.html to browse."
