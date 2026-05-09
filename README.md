# lenia-world-model-project
code related to seminar-project in "Mathematics and AI Internship" module of Data Science course at Uni Leipzig

### 1. Branching Strategy
- **main**: Contains the stable, production-ready code.
- **develop**: The integration branch for features. All pull requests should target this branch.

### 2. Branch Protection Rules
To prevent accidental pushes to protected branches, the following rules are applied to both `main` and `develop`:
- **Require a pull request before merging**: No direct pushes allowed.
- **Require approvals**: At least one team member must review the code (optional but recommended).
- **Do not allow bypassing**: Admins are also subject to these rules.

- ## Development Workflow
please follow this standard workflow for all contributions.

### 1. Sync Local Environment
Before starting any new work, ensure your local `develop` branch is up to date:
```bash
git checkout develop && git pull origin develop
```

### 2. Create a Feature Branch
Create a new branch for your specific task or feature:
```bash
git checkout -b feature/your-feature-name
```

### 3. Development & Commits
Make your changes and commit them with descriptive messages:
```bash
git add .
git commit -m "Brief description of what you did"
```

### 4. Push to GitHub
Push your local branch to the remote repository:
```bash
git push origin feature/your-feature-name
```

### 5. Open a Pull Request (PR)

  Go to the repository on GitHub.

  Click the yellow "Compare & pull request" button that appears at the top.

  The target branch is set to develop by default.

  Add a short description of your changes and submit the PR for review.
