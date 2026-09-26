# Security

Sentinel is a prototype built for the ABB Accelerator. I've hardened the parts I could test, and I've tried to be plain about what I haven't done.

## What it does to limit risk

**No language model in the decision path.** Nothing in `sentinel/` or `api/` calls a generative model, so there's no prompt to inject into and no model output to trust. A test (`tests/test_pipeline.py::test_no_llm_imports_anywhere_in_the_package`) fails the build if a language-model client is imported.

**Bounded file access and uploads.**
- Dataset names and run IDs are reduced to a base name, resolved, and checked to be inside the uploads, samples or runs folder (`is_relative_to`), so `../` tricks return a 404.
- Uploads are capped at 50 MB, limited to `.csv` and `.txt`, given a sanitized file name, and rejected with a 400 if they don't parse as CSV.

**No external calls at run time.** The fonts (IBM Plex, SIL OFL), stylesheets and scripts are bundled in `web/`, so the web app never contacts a CDN. Once the images and data are pulled, it runs without Internet access.

**An optional Docker socket.** The Deploy step builds and smoke-tests the generated model container, which needs `/var/run/docker.sock`. Anything with that socket has root-equivalent access to the host, so in `docker-compose.yml` it's a variable: set `DOCKER_SOCK=/dev/null` and the Deploy step reports that Docker is unavailable, skips the smoke test and still writes the service package.

## What it doesn't do

- Nobody has assessed Sentinel against IEC 62443 or any other security standard, and I make no compliance claim.
- The app container runs as a dedicated non-root user (UID 10001), but it does not implement fine-grained role-based access control (RBAC).
- The demo dataset files and the generated models do not use hardware cryptographic keys.

## Reporting a problem

If you find a vulnerability, open a GitHub issue on [CoderJT-Elite/sentinel-ml](https://github.com/CoderJT-Elite/sentinel-ml/issues) that says you found one, and leave the details out. I'll get in touch to collect them. Please don't post a working exploit in public.
