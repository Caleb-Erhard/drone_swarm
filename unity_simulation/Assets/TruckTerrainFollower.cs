using UnityEngine;

public class TruckTerrainFollower : MonoBehaviour
{
    private const float MinimumOffset = 0.05f;

    private Terrain cachedTerrain;
    private float heightOffset = MinimumOffset;

    public void Configure(float offset)
    {
        cachedTerrain = Terrain.activeTerrain ?? FindFirstObjectByType<Terrain>();
        heightOffset = Mathf.Max(MinimumOffset, offset);
    }

    private void LateUpdate()
    {
        if (cachedTerrain == null)
        {
            cachedTerrain = Terrain.activeTerrain ?? FindFirstObjectByType<Terrain>();
            if (cachedTerrain == null)
            {
                return;
            }
        }

        Vector3 position = transform.position;
        position.y = cachedTerrain.SampleHeight(position) + heightOffset;
        transform.position = position;
    }
}
