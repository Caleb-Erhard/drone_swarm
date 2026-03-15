# Drone Coverage Training Setup

## 1. Install ML-Agents
1. In Unity, open `Window > Package Manager`.
2. Install `ML Agents` (`com.unity.ml-agents`).
3. In your Python environment, install the trainer CLI:
   - `pip install mlagents`

## 2. Scene Wiring
1. Create a `SearchZone` GameObject with a `BoxCollider`.
   - Set collider size to your search area.
   - Enable `Is Trigger`.
2. Create an empty `CoverageTracker` GameObject and add `AreaCoverageTracker`.
   - Assign `Search Zone` to the `SearchZone` collider.
   - Tune `Cell Size` (start with `1.0`) and `Sensor Radius` (start with `2.5`).
3. On your drone GameObject add:
   - `LockedAltitudeDroneController` (existing movement script).
   - `DroneCoverageAgent`.
   - `Behavior Parameters`.
   - `Decision Requester`.
4. In `DroneCoverageAgent`:
   - Assign `Drone Controller` (self).
   - Assign `Coverage Tracker`.
   - Keep `Use Fixed Zone Size` enabled.
   - Set `Fixed Zone Size` to `250 x 200`.
   - For boundary entry behavior, keep `Spawn At Zone Edge` enabled (default).
5. In `Behavior Parameters`:
   - `Behavior Name`: `DroneCoverage`
   - `Vector Observation Size`: `57`
   - `Max Step`: `4000`
   - `Space Size`: `3` continuous actions
   - `Behavior Type`: `Default` for training, `Inference Only` after exporting ONNX
6. In `Decision Requester`:
   - `Decision Period`: `1`

## 3. Action Mapping
- `Action[0]`: strafe left/right
- `Action[1]`: forward/back
- `Action[2]`: yaw turn

## 4. Start Training
Run from project root:

```bash
mlagents-learn Assets/ML-Agents/Configs/drone_coverage_ppo.yaml --run-id=drone_coverage_v1 --force
```

Then press Play in Unity.

## 5. Add Manual Demonstrations (Behavior Cloning Warm Start)
1. Stop trainer and Play mode.
2. On the drone agent, add `Demonstration Recorder`.
3. In `Behavior Parameters`, set `Behavior Type` to `Heuristic Only`.
4. Set recorder output to `Assets/ML-Agents/Demonstrations/droneexpert.demo`.
5. Press Play and fly multiple full sweeps using `WASD` + `Q/E`.
6. Stop Play mode to save the `.demo` file.
7. Set `Behavior Type` back to `Default`.
8. Train with BC + PPO:

```bash
mlagents-learn Assets/ML-Agents/Configs/drone_coverage_ppo_bc.yaml --run-id=drone_coverage_bc_v1 --force
```

Notes:
- Unity may auto-create suffixed files like `droneexpert_0.demo`, `droneexpert_1.demo` when a base name already exists.
- Re-record demos after observation-space changes before re-enabling `demo_path` in `drone_coverage_ppo_bc.yaml`.
- Current BC config points to: `Assets/ML-Agents/Demonstrations/droneexpert_0.demo`.

## 6. What To Watch
- Scene/Game view with `Gizmos` enabled: visited cells and current circular sensor footprint are drawn by `AreaCoverageTracker`.
- TensorBoard stats:
  - `Coverage/Ratio`
  - `Coverage/NewFraction`
  - `Coverage/OutsideDistance`
  - `Coverage/FootprintOverlap01`
  - `Coverage/FrontierAlignment`

## 7. First Tuning Pass
- If the drone spins or stalls:
  - Lower `turnSpeed` and/or `maxSpeed`.
  - Increase `turnInputPenalty` / `yawRatePenalty` in `DroneCoverageAgent`.
- If it hugs edges:
  - Increase `outsideZonePenalty`.
- If it learns slowly:
  - Increase `newCoverageReward` slightly (for example `6 -> 8`).
- If it loops over old ground:
  - Increase `overlapPenalty`, `revisitingPenalty`, and `frontierDistanceRewardScale`.

## 8. Fixed Search Zone (Recommended)
- Train on a single `250 x 200` zone (max swarm-size partition size).
- Keep `Randomize Zone Size Each Episode` and `Use Zone Size Curriculum` disabled while using fixed-size training.
- Keep `Max Expected Zone Dimension` at or above `250`.
- Retrain from scratch whenever observation size or reward terms change.

## 9. Curriculum/Memory Notes
- Both PPO configs use recurrent memory (`sequence_length: 64`, `memory_size: 128`) and a longer horizon (`time_horizon: 256`).
- Built-in curriculum in `DroneCoverageAgent` ramps zone size by episode count.
- Optional external override from trainer config is supported using environment parameters:
  - `difficulty` (0 to 1)
  - `zone_width_min`, `zone_width_max`
  - `zone_depth_min`, `zone_depth_max`
