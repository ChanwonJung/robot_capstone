---
name: github-repo-scope
description: >
  Enforce repo-scoped file editing discipline when Claude is working inside a GitHub repository.
  Use this skill whenever the user asks Claude to edit, create, delete, or modify files
  in a context that includes a git repo — e.g. tasks involving source code, ROS 2 workspaces,
  CMakeLists, package.xml, config files, launch files, URDFs, or any project with a detectable
  .git directory. Also trigger when the user shares a file path that appears to be inside a
  repo (e.g. ~/turtle_ws/src/..., ~/ros2_ws/..., or any path containing a known project name).
  The skill ensures Claude does not silently modify files outside the repo boundary, and forces
  explicit justification before doing so.
---

# GitHub Repo Scope Discipline

## Purpose

When editing files within a GitHub repository, Claude must:

1. **Stay within repo boundaries by default** — only touch files that are inside (or logically part of) the current repo.
2. **Justify out-of-scope edits before executing them** — if a change outside the repo is truly necessary, explain *why* first and get implicit or explicit acknowledgment before proceeding.
3. **Never silently modify system files, global configs, or other repos** as a side effect of a task.

---

## Detecting Repo Context

Claude is in a repo-scoped context if any of the following are true:

- The user shares a file path containing a recognizable workspace pattern:
  - `~/*/ws/`, `~/ros2_ws/`, `~/turtle_ws/`, `~/<project>/src/`, etc.
- The user mentions a repo name, branch, or references a `git clone`, `colcon build`, `CMakeLists.txt`, `package.xml`, or `setup.py`.
- The task involves editing source files (`.cpp`, `.py`, `.hpp`, `.yaml`, `.launch.py`, `.urdf`, `.xacro`, `.json`, `.toml`, etc.) in any structured project directory.
- A `.git/` directory can be inferred or is visible in the path.

---

## Rules of Engagement

### Rule 1 — Default to In-Repo Edits Only

All file reads, writes, creates, and deletes must target paths **inside** the repo root unless Rule 2 applies.

**In-scope examples:**
- `~/turtle_ws/src/my_pkg/CMakeLists.txt`
- `~/capstone_ws/src/arm_description/urdf/arm.urdf.xacro`
- `./config/params.yaml`

**Out-of-scope examples (require Rule 2):**
- `/etc/udev/rules.d/99-robot.rules`
- `~/.bashrc`, `~/.zshrc`
- `/opt/ros/jazzy/...`
- Another repo at a sibling path (e.g., `~/other_project/`)

---

### Rule 2 — Justify Before Touching Out-of-Scope Files

If an out-of-scope file change is **genuinely necessary** (not just convenient), Claude must:

1. **Pause** — do not make the edit yet.
2. **Explain** in a clearly marked block:
   - Which file will be modified
   - Why it cannot be handled inside the repo
   - What the exact change will be
   - Any risks or side effects
3. **Then proceed** — either after explicit user approval, or if the user's instruction already clearly implies it (e.g., "also update my bashrc").

**Format for the justification block:**

```
⚠️ Out-of-Repo Edit Required
File   : <absolute path>
Reason : <why this is necessary and cannot be done inside the repo>
Change : <concise description of what will be modified>
Risk   : <any side effects or caveats>
```

---

### Rule 3 — Never Silently Cascade

Do not edit out-of-scope files as an undeclared side effect. Examples of violations:

- Installing a system package and also editing `/etc/ld.so.conf` without disclosure.
- Adding a ROS source command to `~/.bashrc` as a "helpful bonus" mid-task.
- Modifying a shared CMake config in `/opt/ros/` to unblock a build error.

If a cascade is needed, apply Rule 2 for each out-of-scope file, separately.

---

### Rule 4 — Announce Repo Root Assumption

If the repo root is ambiguous (e.g., the user shares a deep nested path), Claude must state what it assumes the repo root to be before editing:

```
📁 Assumed repo root: ~/turtle_ws/src/my_package/
   (If this is wrong, clarify before I proceed.)
```

---

## Quick Reference

| Situation | Action |
|---|---|
| Editing file clearly inside repo | Proceed directly |
| Repo root is ambiguous | State assumption first |
| File is outside repo but needed | Apply Rule 2 justification block |
| File is outside repo and not needed | Do not touch it |
| Multiple out-of-scope files needed | Justify each one separately |
| User explicitly pre-approves out-of-scope edit | Acknowledge and proceed; still log the path |

---

## Notes for ROS 2 / Robotics Contexts

Common out-of-scope files that often appear in robotics tasks — always justify these:

- `~/.bashrc` / `~/.zshrc` (sourcing ROS, setting `RMW_IMPLEMENTATION`, etc.)
- `/etc/udev/rules.d/` (USB device permissions)
- `/opt/ros/<distro>/` (never modify without strong reason)
- `~/.colcon/` (build defaults)
- System-wide Python packages installed via `sudo pip`

Prefer in-repo alternatives when possible:
- Use a `setup.sh` or `env.sh` inside the repo instead of modifying `~/.bashrc`.
- Use `colcon.meta` or `CMakeUserPresets.json` inside the repo instead of global colcon defaults.
