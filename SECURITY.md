# Security

Sentinel is a prototype built for the ABB Accelerator. I've hardened the parts I could test, and I've tried to be plain about what I haven't done.

## What it does to limit risk

**No language model in the decision path.** Nothing in `sentinel/` or `api/` calls a generative model, so there's no prompt to inject into and no model output to trust. A test (`tests/test_pipeline.py::test_no_llm_imports_anywhere_in_the_package`) fails the build if a language-model client is imported.

**Bounded file access and uploads.**
- Dataset names and run IDs are reduced to a base name, resolved, and checked to be inside the uploads, samples or runs folder (`is_relative_to`), so `../` tricks return a 404.
- Uploads are capped at 50 MB, limited to `.csv` and `.txt`, given a sanitized file name, and rejected with a 400 if they don't parse as CSV.

**No external calls at run time.** The fonts (IBM Plex, SIL OFL), stylesheets and scripts are bundled in `web/`, so the web app never contacts a CDN. Once the images and data are pulled, it runs without Internet access.

**Unprivileged containers.** The app and MLflow containers run as a normal user (UID 1000), not as root.

**The Docker socket is off by default.** The Deploy step can build and smoke-test the generated model container, but that needs `/var/run/docker.sock`, and anything with that socket has root-equivalent access to the host. So `docker-compose.yml` mounts `/dev/null` in its place and the Deploy step reports that Docker is unavailable, skips the smoke test and still writes the service and runs the parity test. To turn the smoke test on:

```bash
DOCKER_SOCK=/var/run/docker.sock docker compose -f docker-compose.yml -f docker-compose.smoke.yml up --build
```

The override runs the app as root so it can reach the socket. Use it on a machine you trust.

**CORS and batch limits.** The API accepts cross-origin calls only from localhost and the hosted demo, and `/predict` rejects batches over 10,000 rows.

## What it doesn't do

- Nobody has assessed Sentinel against IEC 62443 or any other security standard, and I make no compliance claim.
- There's no authentication or user roles. The API and UI assume a trusted network.
- The demo dataset files and the generated models aren't signed.

## Reporting a problem

If you find a vulnerability, open a GitHub issue on [CoderJT-Elite/sentinel-ml](https://github.com/CoderJT-Elite/sentinel-ml/issues) that says you found one, and leave the details out. I'll get in touch to collect them. Please don't post a working exploit in public.
