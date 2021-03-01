using UnityEngine;
using System.Collections;

public class practicebackground : MonoBehaviour {

    public Texture background;

    void OnGUI()
    {
        GUI.depth = 0;
        GUI.DrawTexture(new Rect(Screen.width*0.25f, Screen.height*0.25f, Screen.width*0.5f, Screen.height*0.5f), background);
    }
}
