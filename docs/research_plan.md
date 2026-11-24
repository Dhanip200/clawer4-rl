# Research plan: blind all-terrain walking for CLAWER-4

*Started 20 Nov 2026. Author: Dhanip200.*

## 1. The question (one sentence)

> **Can a cheap 12-servo quadruped learn to walk over rough ground, slopes and small steps using only what it can "feel"
> (servo angle, speed and load + an IMU), with no camera or lidar, and how much does the servo load feedback help?**

"Blind" walking like this is how big research robots (ANYmal, Unitree) cross rough ground. Our twist: doing it on a
**1.7 kg robot with hobby bus servos and an ESP32**, where the sensors are coarse and slow.

## 2. What the robot can feel

### Servo feedback (ST3020, checked in the SCServo library `SMS_STS.h`)
One `FeedBack(id)` call reads all of these from a servo at once (registers 56 to 70):

| Reading | Function | Use for terrain |
|---|---|---|
| Position (4096 steps/turn) | `ReadPos` | Joint angles (the policy already uses them) |
| Speed | `ReadSpeed` | Joint speeds (already used) |
| **Load** (0 to 1000, % of motor drive) | `ReadLoad` | **Is this foot on the ground? Is the leg blocked by a step?** |
| **Current** | `ReadCurrent` | Same as load, plus energy use |
| Voltage, temperature | `ReadVoltage`, `ReadTemper` | Safety only (low battery, overheating) |

### IMU
The Waveshare General Driver board **already has an IMU built in: the QMI8658** (I2C SDA 32 / SCL 33). It gives the same
tilt and spin readings as an MPU6050.
- If you still want the MPU6050, it can share the same I2C pins (address 0x68, no clash with the QMI8658 at 0x6B).
- Pick **one** IMU for the policy and use the same one in every test.

## 3. Hypotheses (what we expect, written down *before* testing)

- **H1:** A policy trained on mixed terrain crosses rough ground and small steps much more often than the current flat-ground policy.
- **H2:** Adding servo **load** to the observations raises the step-climbing success rate, because the robot can feel when a leg hits a step.
- **H3:** A short **history** of past readings (the last ~10 steps) helps more than a single reading, because the robot can work out
  what the ground is like from how its legs have been moving.
- **H4:** Not all randomisation matters. Servo strength and delay matter more for sim-to-real than mass or friction.

## 4. Honest limits

- Body height is about **10 cm**, so normal house stairs (about **17 cm**) are taller than the robot's legs can reach.
  The target is **small steps of 2 to 6 cm** (books, door sills, kerbs) and the **highest step it can climb** becomes a result.
- Servo load is coarse and noisy, so it has to be measured on the real robot before it is trusted in the sim.
- Reading 12 servos must fit inside the 20 ms control step. Not measured yet (`firmware/hw_check` prints it).

## 5. Method

### 5.1 Terrain in simulation (MuJoCo heightfield)
Flat → small random bumps → slopes (5 to 20°) → single steps (1 to 8 cm) → stair runs.
**Curriculum:** each robot starts on easy ground and moves to harder ground when it walks well, and back down when it falls.

### 5.2 Policies to compare (the experiments)

| Name | Trained on | Sees | Purpose |
|---|---|---|---|
| **B0** sine trot | n/a | nothing | Hand-made baseline |
| **B1** flat policy | flat | angles, speeds, IMU | The policy we already have |
| **P1** terrain | terrain | angles, speeds, IMU | Tests H1 |
| **P2** terrain + load | terrain | P1 + servo load | Tests H2 |
| **P3** terrain + load + history | terrain | P2 over the last 10 steps | Tests H3 |
| *(optional)* **P4** teacher-student | terrain | teacher sees the ground; student copies it from P3's inputs | Stretch goal |

Randomisation ablation for H4: train P2 with one randomisation switched off each time (mass, friction, servo strength, delay, sensor noise).

### 5.3 Making load realistic in the sim
In MuJoCo, load ≈ `actuator_force / max_torque × 1000`. On the real robot: hold a leg still, hang known weights on it,
and log `ReadLoad`. Fit that line and add the measured noise to the sim.

## 6. How we measure (same for every policy)

**Sim:** 1,000 runs per terrain level.
**Real:** a fixed obstacle course, **10 tries each**, filmed:

| Test | Levels |
|---|---|
| Flat floor, carpet, grass | speed and falls |
| Slope (board + books) | 5°, 10°, 15°, 20° |
| Single step (books) | 2, 4, 6 cm (higher if it manages) |
| Rough patch (small stones / cables) | cross or not |

**Numbers recorded:**
1. **Success rate**: % of tries that cross without falling
2. **Highest step** climbed with ≥ 80% success
3. **Speed** compared with the command
4. **Energy**: average servo current (lower = better)
5. **Sim-to-real gap**: sim success minus real success

## 7. Steps (one small step at a time)

| # | Step | Status |
|---|---|---|
| 1 | Run `hw_check` on the real robot: servo IDs, load readings, IMU, read time | ⏸️ waiting for the robot |
| 2 | Measure the real robot: weight, joint limits, servo directions, load-vs-weight line | ⬜ |
| 3 | Retrain B1 (the flat policy was lost) and save it | ✅ 21 Nov: 50M steps, 54.7 min on a T4, final score 44.4, walks 0.196 m/s for a 0.2 command (1.96 m in 10 s, 0 m sideways). Brain: `clawer4_policy.params`, video: `media/walk_b1.mp4` |
| 4 | Add terrain + curriculum to the sim | ✅ 22 Nov: env `Clawer4Walk(terrain=True)` starts robots on tile start patches (random kind, random facing), levels up/down at episode end, foot contact + fall check now measured from the ground under the robot. `tools/test_terrain_env.py` passes. Colab practice run OK 22 Nov (score 4.4 → 11.3, robot stands; files in `final-quad-rl\p1-practice-221126`) |
| 5 | Add load (+ noise) to the sim, train P1, P2, P3 | 🟡 P1 run 1 (22 Nov) lost: learned well (score 33.3 at 13.9M) but the Colab tab disconnected at 11:41 PM and never reconnected, nothing downloaded. Re-run 23 Nov with Drive saving |
| 6 | Sim results table + graphs | ⬜ |
| 7 | Run the policy on the ESP32 (legs in the air first) | ⬜ |
| 8 | Real obstacle course, all policies | ⬜ |
| 9 | Write-up: a 4-page workshop-style paper + video + GitHub | ⬜ |

## ▶️ Next session (start here)

**Where we are (24 Nov, 02:20):**
- **P1 run #2 (Kaggle interactive session) was lost.** Scores 4.4, 13.3, 21.0, 26.6, 31.7, 33.8, 36.6, 38.2 (0 to 19.5M steps);
  last line seen 21.6M, nothing downloaded. Average level stayed ~0.1-0.36 / 5. A Colab copy was lost at 4.6M.
- **Level rule fixed (`f6e4c5d`).** The old rule compared straight-line distance from the start with the total distance commanded,
  so robots told to turn failed even when they obeyed. New rule: progress along the commanded direction, and the robot must get
  0.6 m from its start. Simulated perfect robot, up / down per episode: old 14.6% / 39.5%, new 83.6% / 0%.
  Real B1 brain, full physics, level 0, 16 robots, same episodes: old 1 up / 6 down, new 8 up / 0 down.
- Runs #1 and #2 used the old rule, so their levels and scores are not comparable with later runs.
- Servo load (`--load`, P2) and history (`--history 10`, P3) are coded and tested on the CPU, not trained.
- Kaggle Commit practice run (version 2) started 24 Nov ~02:00, to check that files really end up in Output.

**Real speed:** rough-ground training is ~**0.1M steps/min** on a T4 (both Kaggle and Colab), so a 50M run takes
**~7.5 h**. (The earlier "0.5M steps/min, 1 h 50 min" figure was wrong. Flat B1 runs ~0.9M/min.)

**What went wrong (lessons):**
- `files.download` needs the browser tab connected; each download also opens a Chrome "Save as" window that freezes the tab until someone clicks Save.
- The laptop must not sleep (sleep when plugged in = Never, done 23 Nov).
- Never run the training cell twice in one Colab session (it crashes); restart the session first.
- Average-level print fixed (`3096d6f`: divided by episode length).
- Colab free GPU runs out after a few long runs (resets in ~12-24 h). Colab also disconnects long runs.
- Google Drive mount needs the user to click the Google sign-in pop-up (Claude's Chrome control cannot reach it).
- Kaggle: "Save & Run All (Commit)" can sit Queued for hours; the interactive session gets a GPU at once.
  (Commit version 1 actually ended after 1 min 17 s on 23 Nov, it had the old jax[cuda12] install cell.)
  Don't `pip install jax[cuda12]` on Kaggle (clashes with its jax-cuda13 plugin, fixed in `11f163a`).

- **Interactive Kaggle sessions are not safe for long runs:** Kaggle asks "Are you still there?" after 40 min without edits, and
  the files are lost when the session ends. Use Save & Run All (Commit), and don't leave a sleep cell in the notebook.

**Next step (24 Nov):**
1. Check the practice Commit really saved `practice.params` and the videos in Output.
2. Full P1 as a Commit with the fixed level rule (~7.5 h, Kaggle's limit is 12 h). Check that the average level climbs.
3. Download into `final-quad-rl\p1-241126`, copy the brain into `clawer4-rl/`, watch the bumps and stairs videos.

**Then:** train **P2** (load is already in the sim).

**Plan until the robot arrives:**

| Day | Task |
|---|---|
| Sat 21 | Rough ground + levels in training (above) |
| Sun 22 | Add servo **load** to the sim (with noise) |
| Mon 23 | Train P1 (rough ground): run #2 lost, redo on 24 Nov as a Kaggle Commit |
| Tue 24 | Check P1 videos; servo load in the sim DONE 24 Nov (`ae16d5c`, `--load`); P2 (~7.5 h) after P1 is checked |
| Wed 25 | Train P3 (+ history), if time. Code DONE 24 Nov (`--history 10`) |
| Thu 26 | Test all policies in sim: results table + graphs |
| Fri 27 | Read Rudin 2022 (simple summary) |
| Sat 28 | Real-robot prep: ESP32 brain program, obstacle course plan |
| 29+ | Robot: run `hw_check`, measure it, then legs-in-the-air test |

**Colab tips (learned the hard way):**
- Open the notebook from the "Open in Colab" badge (it clones the GitHub repo, so **push code changes first**).
- Keep the laptop plugged in, awake and the tab open for the whole run.
- Chrome asks where to save each file. Save into `OneDrive\...\Documents\final-quad-rl\<run name>\`.
- `train.py` now saves a backup brain at every score check and downloads it every ~10 min. If downloads don't appear,
  use the Files panel (folder icon on the left) → right-click the file → Download, **before** closing the session.

**Not on GitHub yet:** the terrain files, this plan, `clawer4_policy.params`, `media/walk_b1.mp4`.
They must be pushed before Colab can train on rough ground.

## 8. Related work to read (one at a time, later)
- Lee et al. 2020, *Learning quadrupedal locomotion over challenging terrain* (Science Robotics): blind walking with a teacher-student setup.
- Rudin et al. 2022, *Learning to walk in minutes using massively parallel deep RL*: the terrain curriculum we copy.
- Kumar et al. 2021, *RMA: Rapid Motor Adaptation*: learning from a history of readings.
