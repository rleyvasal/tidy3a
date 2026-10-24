# Releasing tidy3A

The release workflow is `.github/workflows/release.yml`. Publishing a GitHub
Release starts it. It requires a tag matching the package version, runs the
reusable CI checks on that commit, then publishes the same source archive
and wheel that passed metadata and clean-install checks. The publishing job
downloads those artifacts without rebuilding them.

## One-time setup

In `rleyvasal/tidy3a`, create a GitHub environment named `pypi`. Configure
its release restrictions and required reviewers in the repository settings.
The workflow uses this environment for the publishing job; naming it in
YAML does not itself configure approval rules.

Register a PyPI trusted publisher with these values:

| Field | Value |
|---|---|
| PyPI project | `tidy3a` |
| GitHub owner | `rleyvasal` |
| GitHub repository | `tidy3a` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

For a new project, use a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
For an existing project you manage, [add the publisher to that project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
The workflow uses [trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
with a short-lived identity token; no saved PyPI API token is required.

The external GitHub environment and PyPI publisher setup have not been
verified by local workflow validation.

## Prepare and publish a version

1. Set `src/tidy3a/__version__.py` to the release version, for example
   `0.1.0`. The workflow rejects development versions such as `0.1.0.dev0`.
2. Move the release's changelog entries under a version-and-date heading.
   Update the README and first-release notes for the published version.
3. Run `scripts/check.sh`. Run the optional real-data examples with their
   datasets available when validating changes to those workflows.
4. Commit the release changes and push them. Create the corresponding
   version tag, for example `v0.1.0`, on that commit and push the tag.
5. Publish a GitHub Release for the tag. This triggers the version check,
   Python 3.11–3.14 tests, minimum-dependency tests, and distribution checks.
6. Approve the `pypi` environment deployment if its configured rules require
   approval. Confirm the publishing job succeeds and verify the version
   listed on [PyPI](https://pypi.org/project/tidy3a/).

If a check fails, the publishing job does not run. Fix the failure before
releasing. After a successful upload, PyPI does not allow replacing that
version's files; subsequent changes need a new version.
