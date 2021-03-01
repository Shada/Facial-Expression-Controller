using System;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using System.IO;

namespace UnityStandardAssets._2D
{
    public class Restarter : MonoBehaviour
    {
        public bool exiter;
        public Texture winnerbox;
        private bool showWinrar = false;
        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.tag == "Player" && other.isTrigger)
            {
                if (exiter)
                {
                    PlatformerCharacter2D player = other.gameObject.GetComponent<PlatformerCharacter2D>();
                    player.WriteToFile();
                    showWinrar = true;
                }
                else
                {
                    PlatformerCharacter2D player = other.gameObject.GetComponent<PlatformerCharacter2D>();
                    player.ResetJumpsAndFlips();
                }
                // save to file
                // show "you are winrar"
            }
        }

        void OnGUI()
        {
            if(showWinrar)
                GUI.DrawTexture(new Rect(Screen.width * 0.25f, Screen.height * 0.4f, Screen.width * 0.5f, Screen.height * 0.2f), winnerbox);
        }
    }
}
