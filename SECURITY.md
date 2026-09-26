# Security Policy

Sentinel is designed for industrial operational technology (OT) and enterprise predictive maintenance environments. Because reliability and safety depend on deterministic execution, security considerations are integrated into the architecture.

---

## 1. Supported Versions

| Version | Supported | Security Maintenance |
|:---|:---:|:---|
| 0.1.x (Prototype Phase) | :white_check_mark: | Active vulnerability monitoring & patch updates |

---

## 2. Threat Model & Architectural Guarantees

Sentinel operates according to the following security design principles:

### A. Zero Generative LLM Execution (AST-Verified)
- **Constraint**: No language model, generative API, external prompt runtime, or stochastic inference is present in Sentinel's decision path.
- **Security Implication**: Immune to prompt injection, jailbreaking, model extraction via chat interfaces, hallucinated maintenance recommendations, and supply-chain exfiltration via LLM orchestration libraries.
- **Enforcement**: Continuous automated AST import audits (`tests/test_pipeline.py::test_no_llm_imports_anywhere_in_the_package`).

### B. Offline operation
- **Local assets**: web fonts (IBM Plex under SIL OFL), stylesheets, icons and scripts are bundled in `web/`. The web app makes no requests to public CDNs (Google Fonts, unpkg, cdnjs).
- **No standards claim**: Sentinel is a prototype and has not been assessed against IEC 62443 or any other security standard.

### C. Docker socket
- The Deploy step builds and smoke-tests the generated model container. That needs the Docker socket (`/var/run/docker.sock`), which gives a process root-equivalent access to the host, so treat the app container accordingly.
- Set `DOCKER_SOCK=/dev/null` to mount nothing useful: the Deploy step reports that Docker is unavailable, skips the smoke test and still writes the service package.

### D. Ingestion & Upload Protections
- **Path Traversal Protection**: All uploaded dataset filenames and run IDs are sanitized with strict base stem extraction (`pathlib.Path.name`) and validated against directory canonical boundaries (`p.is_relative_to(uploads_dir)`).
- **Payload Limits**: Uploaded CSV files are capped at 50 MB to prevent resource exhaustion and denial-of-service.
- **Format Validation**: File extensions are restricted to `.csv` and `.txt`, and tabular structures are validated prior to pipeline execution.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability or supply chain issue in Sentinel:

1. **Do NOT open a public GitHub issue.**
2. Send a confidential report to the project maintainer:
   - **Contact**: John Tewolde (`john@sentinel-ml.org` / GitHub: `@CoderJT-Elite`)
3. Include:
   - Description of the vulnerability and attack vector.
   - Minimal reproducible proof-of-concept (PoC).
   - Component affected (`api/`, `sentinel/`, `web/`, or dependencies).
   - Potential impact on industrial plant assets or determinism.

We acknowledge receipt of reports within 24 hours and aim to release verified patches within 72 hours.
