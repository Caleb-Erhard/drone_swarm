using UnityEngine;

[DisallowMultipleComponent]
public class DroneDemoZoneVisualizer : MonoBehaviour
{
    [SerializeField] private Vector2 zoneCenterXZ;
    [SerializeField] private Vector2 zoneSizeXZ = new Vector2(250f, 200f);
    [SerializeField] private float gizmoHeight = 200f;
    [SerializeField] private Color zoneColor = new Color(0f, 0.8f, 1f, 0.9f);

    public void Configure(Vector2 centerXZ, Vector2 sizeXZ, Color color)
    {
        zoneCenterXZ = centerXZ;
        zoneSizeXZ = sizeXZ;
        zoneColor = color;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = zoneColor;
        Gizmos.DrawWireCube(
            new Vector3(zoneCenterXZ.x, transform.position.y, zoneCenterXZ.y),
            new Vector3(zoneSizeXZ.x, gizmoHeight, zoneSizeXZ.y));
    }
}
