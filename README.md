# lenia-world-model-project
code related to seminar-project in "Mathematics and AI Internship" module of Data Science course at Uni Leipzig

## Project Structure

```text
lenia-world-models/
├── data/                   # Raw and processed Lenia trajectories (.npy or .h5)
├── models/                 # Neural network architectures
│   ├── encoder.py          # CNN/Transformer backbone
│   ├── decoder.py          # Pixel-prediction decoder (Setup A)
│   ├── predictor.py        # JEPA latent predictor (Setup B)
│   └── world_model.py      # Integrated model logic
├── training/               # Training logic and loops
│   ├── trainer.py          # Training orchestrator
│   └── losses.py           # Loss functions (MSE, VICReg, etc.)
├── scripts/                # Executable scripts
│   ├── generate_data.py    # Lenia simulation generator
│   ├── train_single.py     # Single training execution
│   └── run_hpo.py          # Optuna hyperparameter optimization
├── experiments/            # Checkpoints, logs, and plots
├── utils/                  # Helper functions (visualization, physics)
├── config.yaml             # Global configuration parameters
└── requirements.txt        # Python dependencies
```

## 1. Branching Strategy
- **main**: Contains the stable, production-ready code.
- **develop**: The integration branch for features. All pull requests should target this branch.

## 2. Branch Protection Rules
To prevent accidental pushes to protected branches, the following rules are applied to both `main` and `develop`:
- **Require a pull request before merging**: No direct pushes allowed.
- **Require approvals**: At least one team member must review the code.
- **Do not allow bypassing**: Admins are also subject to these rules.

---

## Synchronization & Conflict Management

To maintain a clean repository and avoid breaking the World Model architecture, always check for remote changes and potential conflicts before starting your work.

### 1. Fetch Remote Changes
Update your local knowledge of the remote repository without affecting your files:
```bash
git fetch origin

```

### 2. Check for Conflicts

See if your local branch has fallen behind the remote branch:

```bash
git status

```

If the output says *"Your branch is behind 'origin/develop' by X commits"*, you must synchronize before proceeding to avoid merge conflicts later.

### 3. Safe Pull & Resolution

Merge the remote changes into your local workspace:

```bash
git pull origin develop

```

*Note: If Git reports a merge conflict, stop and resolve the marked sections in the affected files manually. Once resolved, use `git add` and `git commit` to finalize the merge.*

---

## Development Workflow

Please follow this standard workflow for all contributions.

### 1. Sync Local Environment

Ensure your local `develop` branch is up to date:

```bash
git checkout develop && git pull origin develop

```

### 2. Create a Feature Branch

Create a new branch for your specific task:

```bash
git checkout -b feature/your-feature-name

```

### 3. Development & Commits

Make your changes and commit them with descriptive messages:

```bash
git add .
git commit -m "Brief description of what you did"

```

### 4. Final Conflict Check

Before pushing, it is good practice to pull from `develop` one last time to ensure no new conflicts were introduced while you were coding:

```bash
git pull origin develop

```

### 5. Push to GitHub

Push your local branch to the remote repository:

```bash
git push origin feature/your-feature-name

```

### 6. Open a Pull Request (PR)

1. Go to the repository on GitHub.
2. Click the "Compare & pull request" button.
3. Ensure the target branch is set to `develop`.
4. Add a short description of your changes and submit the PR for review.

```


