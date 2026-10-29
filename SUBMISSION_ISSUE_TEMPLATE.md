# MP2 submission issue template

The MP2 website prepares a GitHub issue after you enter your student ID and score.
Create that issue with your GitHub account; preparing the link alone does not submit
anything. Replace the placeholders below.

```markdown
MP2 submission — <STUDENT ID>

Score: 84.3683
Split: test (467 questions, 394 correct)
Protocol: dase7506-mp2-open-hard-verifier-v5

Code repository:
https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>

Submitted code version:
commit <COMMIT-SHA> (branch main)

Checkpoint:
https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>/tree/main/code/runs/final_bundle
model.npz SHA-256:
39D2478B11B51CAEA2EF7539840B41ECFF1A17E7E83831D9464C8884AE4D5B06

Report:
https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>/blob/main/report.md

Reproduction:
https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>/blob/main/REPRODUCE.md

Official evaluation JSON:
https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>/blob/main/code/runs/final-test-score.json

Hardware used for the timing measurement:
<CPU model>, 4 threads, <RAM> GB, <OS>

AI/reuse disclosure:
AI coding assistance was used to implement and run experiments, refactor the final
code and help draft the report. All design decisions, protocol checks and the final
score were reviewed by the author. No external datasets, pretrained weights or code
beyond the supplied package were used.
```

## Before submitting

1. Fill in your student ID in `README.md`, `report.md` and this issue.
2. Create a **public** GitHub repository and push this directory:
   ```bash
   git remote add origin https://github.com/<YOUR-GITHUB-USER>/<YOUR-REPO>.git
   git branch -M main
   git push -u origin main
   ```
3. Confirm the checkpoint and report links open without authentication.
4. Run the commands in `REPRODUCE.md` from a clean clone and check that the official
   scorer returns `84.36830835117773`.
5. On the MP2 website, enter the student ID and score, click **Submit on GitHub**, and
   create the prepared issue. Use the same GitHub account and student ID for updates.
