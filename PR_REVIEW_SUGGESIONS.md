The Black configuration in setup.cfg will not be picked up by Black (Black reads configuration from pyproject.toml under [tool.black], not from setup.cfg). As written, CI/pre-commit will run Black with its defaults (line length 88, different target versions), which will diverge from the intended 120/py310 settings.

-----------

The Black pre-commit hook does not pass line-length/target-version args, and Black will not read the intended settings from setup.cfg. This will make pre-commit enforce Black defaults instead of the repo's 120/py310 policy.

Suggested changeset 1 (1)
.pre-commit-config.yaml
Original file line number	Diff line number	Diff line change
     hooks:
       - id: black
         language_version: python3.10
       + args: ["--line-length=120", "--target-version=py310"]
   - repo: https://github.com/pycqa/isort
     rev: 5.13.2


------------

CI installs unpinned lint/format tool versions, which can drift from the pinned versions used by pre-commit/requirements.txt and cause inconsistent formatting/lint results over time. Pin these to the same versions used elsewhere in the repo.

Suggested changeset 1 (1)
.github/workflows/ci.yml
Original file line number	Diff line number	Diff line change
           python-version: '3.10'
       - name: Install linting tools
         run: pip install flake8 black isort
         run: pip install flake8==7.1.1 black==24.10.0 isort==5.13.2
       - name: Check formatting (black)
         run: black --check thesis_writing/ thesis_doc/ visualize/


------------

The CI Black check references thesis_writing/, but that directory does not exist in the repository. This will cause CI to fail with "Path ... does not exist". Also, Black won't read setup.cfg, so line-length/target-version need to be passed explicitly (or moved to pyproject.toml).

Suggested changeset 1 (1)
.github/workflows/ci.yml
Original file line number	Diff line number	Diff line change
         run: pip install flake8 black isort
       - name: Check formatting (black)
         run: black --check thesis_writing/ thesis_doc/ visualize/
         run: black --check --line-length 120 --target-version py310 thesis_doc/ visualize/
       - name: Check import order (isort)
         run: isort --check-only thesis_writing/ thesis_doc/ visualize/


------------

The CI isort check references thesis_writing/, but that directory does not exist in the repository, so the step will fail. Point isort at the existing directories (and explicitly pass the config path if you want to ensure it uses setup.cfg).

Suggested changeset 1 (1)
.github/workflows/ci.yml
Original file line number	Diff line number	Diff line change
         run: black --check thesis_writing/ thesis_doc/ visualize/
       - name: Check import order (isort)
         run: isort --check-only thesis_writing/ thesis_doc/ visualize/
         run: isort --check-only --settings-path setup.cfg thesis_doc/ visualize/
       - name: Lint Python files (flake8)
         run: flake8 thesis_writing/ thesis_doc/ visualize/


-------------

The CI flake8 invocation references thesis_writing/, but that directory does not exist in the repository, so the step will fail. Update the paths to only include directories that exist (and optionally pass --config setup.cfg for clarity).

Suggested changeset 1 (1)
.github/workflows/ci.yml
Original file line number	Diff line number	Diff line change
         run: isort --check-only thesis_writing/ thesis_doc/ visualize/
       - name: Lint Python files (flake8)
         run: flake8 thesis_writing/ thesis_doc/ visualize/
         run: flake8 --config setup.cfg thesis_doc/ visualize/
