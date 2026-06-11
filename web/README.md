# torch-volpy API Website

This folder is a static API search site generated from `src/torch_volpy`.

## Refresh the API index

Run from the repository root:

```bash
python web/generate_api_index.py
```

This updates `web/api.json`, which is the searchable data source for both the
browser UI and bots.

## Preview locally

```bash
python -m http.server 8000 --directory web
```

Open `http://localhost:8000`.

## Deploy on GitHub Pages

The workflow at `.github/workflows/deploy-web.yml` publishes this `web` folder.
In the repository settings, set **Pages** to deploy from **GitHub Actions**.
