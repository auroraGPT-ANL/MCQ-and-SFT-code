# A Proposed Approach for the Hackathon

Our team has six people, though 1-2 may join us.  We will have two days with multiple 90-minute sessions each day.
The third day will be a moving hackathon with another 3 sessions.

We propose the following for discussion during our initial session, looking for guidance to adjust these to
leverage insights of from those with more experience doing intensive hackathon sprints.

Here is o3's suggested approach based on a detailed explanaiton of our goals, logistics, group size, and time constraints.


## Sauna-Hack Developer Coordination Guide (prepared by o3)

This guide outlines how we’ll coordinate across 6 developers working on the `sauna-hack` branch during the hackathon.
The goal is **fast progress with minimal collisions or confusion**.

---

### 🔁 1. Shared Task Board (Kanban Style)
- Use a **whiteboard** or GitHub Projects board with columns:
  - `To Do` · `In Progress` · `Done`
- Tasks are written on sticky notes or cards.
- When someone grabs a task, they move it to `In Progress` and add their initials.

### 💬 2. Verbal File Locks (Lightweight)
- For shared files (like `README.md`, `config.yml`, or `agent_runner.py`), simply **announce edits aloud**:
  > “I’m editing `README.md` — give me 5 minutes.”
- For longer edits, mark it on the whiteboard:
  > `README: Charlie (WIP)`

### 🧑‍🤝‍🧑 3. Pairing or Micro-Teams
- Pair up (or form small triads) to tackle hard tasks.
- One codes, one tests, one writes doc = faster results and fewer bugs.
- Great for debugging, prompt crafting, or workflow integration.

### 🔀 4. Short Pull Request Cycle
- Create a PR into `sauna-hack` for **every unit of work**.
- Do **live reviews** (e.g., “Hey, can you approve this?”).
- **Squash merge** to keep commit history clean.

### 🕘 5. Time-Boxed Syncs
- **Start of day:** 5-min standup (who’s doing what).
- **Lunch:** quick re-align if needed.
- **End of day:** recap, blockers, handoffs.

### 📁 6. Strategy for High-Collision Files
- For shared files (e.g., `README.md`, workflow configs):
  - Make short-lived branches.
  - Batch small edits into one PR per person.
  - Or temporarily assign one person as “owner” for the day.

### 🔄 7. Git Tips to Avoid Clobbering Each Other
- Always pull with rebase:
  ```zsh
  git pull --rebase
  ```
- If something gets overwritten: use `git reflog` or GitHub “Revert changes” feature.

---

### ✅ Summary Table

| Practice                     | Benefit                             |
|-----------------------------|-------------------------------------|
| Verbal locks for key files  | Avoid mid-edit collisions           |
| Shared task board           | Keeps everyone aligned              |
| Live PR reviews             | Speed + stability                   |
| Daily syncs                 | Adjust plans & unblock fast         |
| File ownership for shared   | Reduces merge conflicts             |
| Pair programming            | More eyes = fewer bugs              |

---

Stay relaxed, stay respectful, and build awesome stuff together!
