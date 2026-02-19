# Uni-Xervo Website Docs

Documentation site for **Uni-Xervo** using MkDocs + Material.

## Setup

```bash
cd website
poetry install
```

## Local development

```bash
poetry run mkdocs serve
```

Visit: http://localhost:8000

## Build

```bash
poetry run mkdocs build
```

Output is generated in `website/site/`.

## Notes

- Branding and colors come from `docs/assets/stylesheets/theme.css`.
- Site config is in `mkdocs.yml`.
- Content pages live under `docs/`.
