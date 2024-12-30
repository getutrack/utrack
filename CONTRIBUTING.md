# Contributing to Utrack

Thank you for showing an interest in contributing to Utrack! All kinds of contributions are valuable to us. In this guide, we will cover how you can quickly onboard and make your first contribution.

## Submitting an issue

Before submitting a new issue, please search the [issues](https://github.com/getutrack/utrack/issues) tab. Maybe an issue or discussion already exists and might inform you of workarounds. Otherwise, you can give new information.

While we want to fix all the [issues](https://github.com/getutrack/utrack/issues), before fixing a bug we need to be able to reproduce and confirm it. Please provide us with a minimal reproduction scenario using a repository or [Gist](https://gist.github.com/). Having a live, reproducible scenario gives us the information without asking questions back & forth with additional questions like:

- 3rd-party libraries being used and their versions
- a use-case that fails

Without said minimal reproduction, we won't be able to investigate all [issues](https://github.com/getutrack/utrack/issues), and the issue might not be resolved.

You can open a new issue with this [issue form](https://github.com/getutrack/utrack/issues/new).

## Projects setup and Architecture

### Requirements

- Node.js version v16.18.0
- Python version 3.8+
- Postgres version v14
- Redis version v6.2.7

### Setup the project

The project is a monorepo, with backend api and frontend in a single repo.

The backend is a django project which is kept inside apiserver

1. Clone the repo

```bash
git clone https://github.com/getutrack/utrack.git [folder-name]
cd [folder-name]
chmod +x setup.sh
```

2. Run setup.sh

```bash
./setup.sh
```

3. Start the containers

```bash
docker compose -f docker-compose-local.yml up
```
