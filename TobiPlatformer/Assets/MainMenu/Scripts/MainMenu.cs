using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using System.IO;

public class MainMenu : MonoBehaviour {

    public Texture backgroundTexture;

    public int participantID = -1;
    
    public string texttoedit = "";

    public string numbers = "0123456789";

    void OnGUI()
    {
        // Display background texture
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), backgroundTexture);

        texttoedit = GUI.TextField(new Rect(Screen.width * .25f, Screen.height * .5f, Screen.width * .5f, Screen.height * .05f), texttoedit);
        
        if(texttoedit.Length > 0)
        {
            if(!numbers.Contains(texttoedit.Substring(texttoedit.Length - 1)))
            {
                print("Should remove the text, since it is not a number");
                texttoedit = texttoedit.Remove(texttoedit.Length - 1, 1);
            }
            
        }

        // Display buttons
        if (GUI.Button(new Rect(Screen.width * 0.25f, Screen.height * 0.75f, Screen.width * .5f, Screen.height * .1f), "Play Game!"))
        {
            if (texttoedit.Length > 0)
            {
                participantID = int.Parse(texttoedit);
            }
            else
                participantID = -1;

            // Check if ID is valid or not.. 
            if (participantID >= 0)
            {
                    Participant.participantID = participantID;
                    SceneManager.LoadScene("practice");
            }
        }
    }
}
