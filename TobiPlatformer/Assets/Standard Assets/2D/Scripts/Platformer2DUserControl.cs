using System;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;
using UnityEngine.SceneManagement;

namespace UnityStandardAssets._2D
{
    [RequireComponent(typeof (PlatformerCharacter2D))]
    public class Platformer2DUserControl : MonoBehaviour
    {
        private PlatformerCharacter2D m_Character;
        private bool m_Jump;


        private void Awake()
        {
            m_Character = GetComponent<PlatformerCharacter2D>();
        }


        private void Update()
        {
            // Read the jump input in Update so button presses aren't missed.
            if((Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.JoystickButton2))&&!m_Character.finished)
                m_Jump = true;

            if (Input.GetKey(KeyCode.Escape))
                Application.Quit();

            if(Input.GetKeyDown(KeyCode.Return) && SceneManager.GetActiveScene().name == "practice")
                SceneManager.LoadScene("Level-1");

        }


        float h;
        private void FixedUpdate()
        {
            // Read the inputs.
            bool crouch = false;
            if ((Input.GetKey(KeyCode.JoystickButton0) || Input.GetKey(KeyCode.LeftArrow)) && !m_Character.finished)
                h = -1.0f;
            else if ((Input.GetKey(KeyCode.JoystickButton1) || Input.GetKey(KeyCode.RightArrow)) && !m_Character.finished)
                h = 1.0f;
            else
            {
                if (!m_Character.finished)
                    h = 0.0f;
                else
                {
                    h *= 0.95f;
                    if (h <= 0.1f)
                        h = 0.0f;
                }
            }
            // Pass all parameters to the character control script.
            m_Character.Move(h, crouch, m_Jump);
            m_Jump = false;
        }
    }
}
