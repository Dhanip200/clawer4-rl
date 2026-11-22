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
| 4 | Add terrain + curriculum to the sim | ✅ 22 Nov: env `Clawer4Walk(terrain=True)` starts robots on tile start patches (random kind, random facing), levels up/down at episode end, foot contact + fall check now measured from the ground under the robot. `tools/test_terrain_env.py` passes. Colab practice run still to do |
| 5 | Add load (+ noise) to the sim, train P1, P2, P3 | ⬜ |
| 6 | Sim results table + graphs | ⬜ |
| 7 | Run the policy on the ESP32 (legs in the air first) | ⬜ |
| 8 | Real obstacle course, all policies | ⬜ |
| 9 | Write-up: a 4-page workshop-style paper + video + GitHub | ⬜ |

## ▶️ Next session (start here)

**Where we stopped (22 Nov):** rough ground + curriculum are in the env and pass the laptop tests
(`python tools/test_terrain_env.py`: stands on all 5 kinds, ground height matches MuJoCo, levels go up/down, no NaNs).

**Next step:**
1. Push the new files to GitHub (Colab clones the repo).
2. 7-minute Colab practice run on rough ground: `train_policy(terrain=True, num_timesteps=5_000_000, out="clawer4_p1_test.params")`.
   Check: score goes up, no NaNs, "average level" lines appear and creep above 0.

**Then:** train **P1** (rough ground, no load feeling), ~1 hour on Colab.

**Plan until the robot arrives:**

| Day | Task |
|---|---|
| Sat 21 | Rough ground + levels in training (above) |
| Sun 22 | Add servo **load** to the sim (with noise) |
| Mon 23 | Train P1 (rough ground) |
| Tue 24 | Train P2 (rough ground + load) |
| Wed 25 | Train P3 (+ history), if time |
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
