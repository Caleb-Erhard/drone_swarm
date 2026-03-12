#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using Unity.MLAgents.Demonstrations;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Keeps demonstration recording output deterministic:
/// - clears old demo files before entering Play (when Record is enabled)
/// - remaps the recorder's sanitized filename back to the configured name after Play stops
/// </summary>
[InitializeOnLoad]
public static class DemoRecordingFileManager
{
    const string DemoExtension = ".demo";
    const int RecorderNameLimit = 16;
    static readonly Regex InvalidNameRegex = new Regex("[^a-zA-Z0-9 -]", RegexOptions.Compiled);
    static readonly List<PendingRecording> PendingRecordings = new List<PendingRecording>();

    struct PendingRecording
    {
        public string DirectoryPath;
        public string DesiredBaseName;
        public string SanitizedBaseName;
    }

    static DemoRecordingFileManager()
    {
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
    }

    static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.ExitingEditMode)
        {
            PrepareForRecording();
        }
        else if (state == PlayModeStateChange.EnteredEditMode)
        {
            FinalizeRecordedFiles();
        }
    }

    static void PrepareForRecording()
    {
        PendingRecordings.Clear();
        var recorders = UnityEngine.Object.FindObjectsOfType<DemonstrationRecorder>(true);

        foreach (var recorder in recorders)
        {
            if (recorder == null || !recorder.Record)
            {
                continue;
            }

            var desiredBaseName = NormalizeBaseName(recorder.DemonstrationName);
            if (string.IsNullOrWhiteSpace(desiredBaseName))
            {
                Debug.LogWarning("[DemoRecordingFileManager] DemonstrationName is empty; skipping overwrite setup.");
                continue;
            }

            var sanitizedBaseName = SanitizeForRecorder(desiredBaseName);
            if (string.IsNullOrWhiteSpace(sanitizedBaseName))
            {
                Debug.LogWarning("[DemoRecordingFileManager] DemonstrationName becomes empty after sanitization; skipping overwrite setup.");
                continue;
            }

            var directoryPath = ResolveDirectoryPath(recorder.DemonstrationDirectory);
            if (string.IsNullOrWhiteSpace(directoryPath))
            {
                Debug.LogWarning("[DemoRecordingFileManager] DemonstrationDirectory is invalid; skipping overwrite setup.");
                continue;
            }

            Directory.CreateDirectory(directoryPath);
            CleanupOldFiles(directoryPath, desiredBaseName, sanitizedBaseName);

            PendingRecordings.Add(new PendingRecording
            {
                DirectoryPath = directoryPath,
                DesiredBaseName = desiredBaseName,
                SanitizedBaseName = sanitizedBaseName
            });
        }
    }

    static void FinalizeRecordedFiles()
    {
        if (PendingRecordings.Count == 0)
        {
            return;
        }

        var changedFiles = false;
        foreach (var pending in PendingRecordings)
        {
            var newestRecorded = FindNewestRecordedFile(pending.DirectoryPath, pending.SanitizedBaseName);
            if (string.IsNullOrWhiteSpace(newestRecorded))
            {
                continue;
            }

            var desiredPath = Path.Combine(pending.DirectoryPath, pending.DesiredBaseName + DemoExtension);
            if (!PathsEqual(newestRecorded, desiredPath))
            {
                DeleteFileAndMeta(desiredPath);
                MoveFileWithMeta(newestRecorded, desiredPath);
                changedFiles = true;
            }

            foreach (var stale in Directory.GetFiles(pending.DirectoryPath, pending.SanitizedBaseName + "*" + DemoExtension))
            {
                if (!PathsEqual(stale, desiredPath))
                {
                    DeleteFileAndMeta(stale);
                    changedFiles = true;
                }
            }
        }

        PendingRecordings.Clear();
        if (changedFiles)
        {
            AssetDatabase.Refresh();
        }
    }

    static void CleanupOldFiles(string directoryPath, string desiredBaseName, string sanitizedBaseName)
    {
        DeleteFileAndMeta(Path.Combine(directoryPath, desiredBaseName + DemoExtension));
        foreach (var stale in Directory.GetFiles(directoryPath, sanitizedBaseName + "*" + DemoExtension))
        {
            DeleteFileAndMeta(stale);
        }
    }

    static string FindNewestRecordedFile(string directoryPath, string sanitizedBaseName)
    {
        var candidates = Directory.GetFiles(directoryPath, sanitizedBaseName + "*" + DemoExtension);
        if (candidates.Length == 0)
        {
            return null;
        }

        string newestFile = null;
        DateTime newestWrite = DateTime.MinValue;
        foreach (var candidate in candidates)
        {
            var writeTime = File.GetLastWriteTimeUtc(candidate);
            if (writeTime > newestWrite)
            {
                newestWrite = writeTime;
                newestFile = candidate;
            }
        }
        return newestFile;
    }

    static string ResolveDirectoryPath(string directorySetting)
    {
        if (string.IsNullOrWhiteSpace(directorySetting))
        {
            return Path.Combine(Application.dataPath, "Demonstrations");
        }

        if (Path.IsPathRooted(directorySetting))
        {
            return directorySetting;
        }

        return Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), directorySetting));
    }

    static string NormalizeBaseName(string demoName)
    {
        if (string.IsNullOrWhiteSpace(demoName))
        {
            return string.Empty;
        }

        var trimmed = demoName.Trim();
        if (trimmed.EndsWith(DemoExtension, StringComparison.OrdinalIgnoreCase))
        {
            return trimmed.Substring(0, trimmed.Length - DemoExtension.Length);
        }

        return trimmed;
    }

    static string SanitizeForRecorder(string name)
    {
        var cleaned = InvalidNameRegex.Replace(name, "");
        if (cleaned.Length > RecorderNameLimit)
        {
            cleaned = cleaned.Substring(0, RecorderNameLimit);
        }

        return cleaned;
    }

    static void MoveFileWithMeta(string sourcePath, string destinationPath)
    {
        File.Move(sourcePath, destinationPath);

        var sourceMeta = sourcePath + ".meta";
        var destinationMeta = destinationPath + ".meta";
        if (File.Exists(sourceMeta))
        {
            if (File.Exists(destinationMeta))
            {
                File.Delete(destinationMeta);
            }
            File.Move(sourceMeta, destinationMeta);
        }
    }

    static void DeleteFileAndMeta(string filePath)
    {
        if (File.Exists(filePath))
        {
            File.Delete(filePath);
        }

        var metaPath = filePath + ".meta";
        if (File.Exists(metaPath))
        {
            File.Delete(metaPath);
        }
    }

    static bool PathsEqual(string a, string b)
    {
        if (string.IsNullOrWhiteSpace(a) || string.IsNullOrWhiteSpace(b))
        {
            return false;
        }

        var pathA = Path.GetFullPath(a).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var pathB = Path.GetFullPath(b).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return string.Equals(pathA, pathB, StringComparison.OrdinalIgnoreCase);
    }
}
#endif
