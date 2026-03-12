Store recorded expert demonstrations in this folder.

Canonical training demo file:
- `droneexpert_active.demo`

Recorded with Unity `Demonstration Recorder` on the `DroneCoverageAgent` while `Behavior Type` is `Heuristic Only`.

This project includes `Assets/ML-Agents/Editor/DemoRecordingFileManager.cs` to keep demo output clean:
- before Play (when `Record` is on), old demo files are deleted
- after Play, the newest recorded file is renamed to `droneexpert_active.demo`

Result: one demo file is kept and overwritten each recording session.
