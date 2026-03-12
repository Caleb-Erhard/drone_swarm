# Drone ML Training Status (Verified Snapshot)

Last updated from on-disk files: March 9, 2026 (local file timestamps)
Project root: `C:\Users\caleb\My project`

## 1) Canonical files and paths

- Main scene (use this path for team consistency):
  - `Assets/Scenes/SampleScene.unity`
- BC config:
  - `Assets/ML-Agents/Configs/drone_coverage_ppo_bc.yaml`
- PPO config (no BC):
  - `Assets/ML-Agents/Configs/drone_coverage_ppo.yaml`
- Canonical demo file used by BC:
  - `Assets/ML-Agents/Demonstrations/droneexpert_active.demo`

## 2) Current BC config values (verified)

From `Assets/ML-Agents/Configs/drone_coverage_ppo_bc.yaml`:

- `demo_path: Assets/ML-Agents/Demonstrations/droneexpert_active.demo`
- `behavioral_cloning.strength: 0.9`
- `behavioral_cloning.steps: 1200000`
- PPO `max_steps: 3000000`

## 3) Current scene values (verified)

From `Assets/Scenes/SampleScene.unity`:

- Drone controller:
  - `maxSpeed: 14`
  - `acceleration: 8`
  - `turnSpeed: 85`
- Decision requester:
  - `DecisionPeriod: 2`
- Episode limits:
  - `agentParameters.maxStep: 7000`
  - `MaxStep: 7000`
  - `episodeStepLimit: 7000`
- Behavior Parameters:
  - `m_BehaviorType: 0` (Default)
- Demonstration Recorder:
  - `Record: 0` (currently OFF)
  - `NumStepsToRecord: 0` (unlimited)
  - `DemonstrationName: droneexpert_active`
  - `DemonstrationDirectory: Assets/ML-Agents/Demonstrations`

## 4) Demo file state (verified)

Current files under `Assets/ML-Agents/Demonstrations`:

- `droneexpert_active.demo` (exists)
- No other `.demo` files currently present

Last recorded timestamp seen on disk:

- `droneexpert_active.demo` -> `3/9/2026 3:32:37 AM`

Important:

- If you record a new demo, this timestamp and file size should change.
- If timestamp does not change, recording did not save (most common reason: `Record` left OFF).

## 5) Exact workflow for recording new demos

1. Open `Assets/Scenes/SampleScene.unity`.
2. Select the drone agent object with `DroneCoverageAgent`.
3. Set Behavior Parameters `Behavior Type = Heuristic Only`.
4. In Demonstration Recorder:
   - Set `Record = ON`
   - Keep `DemonstrationName = droneexpert_active`
   - Keep `DemonstrationDirectory = Assets/ML-Agents/Demonstrations`
5. Press Play and fly manual episodes.
6. Stop Play.
7. Set `Record = OFF`.
8. Set Behavior Parameters `Behavior Type = Default`.

Verification command after recording:

```powershell
Get-Item "Assets\ML-Agents\Demonstrations\droneexpert_active.demo" | Select-Object Name,Length,LastWriteTime
```

Expected: `LastWriteTime` updates to current time, and usually `Length` changes.

## 6) Exact training command

Run from project root with `.venv-mlagents` activated:

```powershell
mlagents-learn Assets/ML-Agents/Configs/drone_coverage_ppo_bc.yaml --run-id=drone_coverage_bc_v14 --force --time-scale=5 --capture-frame-rate=0 --target-frame-rate=60 --timeout-wait=180
```

Then press Play in Unity.

## 7) Environment state (verified)

In `.venv-mlagents`:

- `mlagents==1.1.0`
- `mlagents-envs==1.1.0`
- `torch==2.2.2`
- `numpy==1.23.5`
- `onnx==1.15.0`
- `protobuf==3.20.3`
- `onnxscript` not installed

`pip check` result: `No broken requirements found.`

## 8) Script changes already in project

`Assets/DroneCoverageAgent.cs` includes:

- Heuristic control scaling fields:
  - `heuristicMoveScale`
  - `heuristicTurnScale`
  - precision modifier key/scale
- Episode outcome stats:
  - `Coverage/EpisodeResult/Completed`
  - `Coverage/EpisodeResult/OutOfBounds`
  - `Coverage/EpisodeResult/Collision`
  - `Coverage/EpisodeResult/Timeout`
  - `Coverage/EpisodeEndCoverage01`

## 9) Backup scenes kept intentionally

These files exist as backups/recovery safety:

- `Assets/Scenes/SampleScene_before_restore_20260309_043539.unity`
- `Assets/Scenes/SampleScene_recovered_from_backup.unity`

Do not delete until training behavior is confirmed stable.

## 10) Known behavior issue to investigate next

Observed in recent runs: high-speed exits/looping and diagonal bias despite improving mean reward.

Potential contributors (currently true in `SampleScene`):

- high speed and acceleration (`14`, `8`)
- slower control update (`DecisionPeriod: 2`)

If needed in next tuning pass, adjust scene values and re-train with a new run-id.
