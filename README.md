# Celeb_Demo_tflite
This demo will create and compare your face embedding to face embeddings you created beforehand. 

**Files:**<br/><pre>
*aidemo.py*       : contains the GUI of the Demo<br/>
*ai.py*           : contains the Neural Network creating the Embeddings. It Returns Euclidian Distance and index of the minimal euclidain distance.<br/>
*cameraimx8mp.py* : contains the gstreamer pipeline to read in the camera image when used on the ARM device<br/>
*camerapc.py*     : contains the opencv pipeline to read in the camera image when used on the x86 device<br/>
*v4l2_2.py*       : contains the video for Linux settings<br/>
</pre>
**Prerequisite:**<br/>
After cloning the git repository please add the model file which you can find [here (13)](https://mega.nz/file/kZkziCqT#hddnG2MoEWf4YqDfQUSyyJgzraCN_Dh-DOsspy5D2zY), [here (15)](https://mega.nz/file/4B0BjKhA#gOoGpmufVrbY3EJ1Zv6Jks9aSKBJnDRZ6w9NbVnVKPQ), [here (15 all int)](https://mega.nz/file/5c1BBaxK#j-CCgjag5hsjoyBh4QYz5rwdq1CLPCTVzxD4WVAS0RY), and [here (220 all int)](https://mega.nz/file/NRshSYaD#j-CCgjag5hsjoyBh4QYz5rwdq1CLPCTVzxD4WVAS0RY) to the demo-data/
models/tflite/ folder<br/>
And add the Embeddingsfiles, which you can find [here (220)](https://mega.nz/file/NZ8D3KLb#xaR7Ke60CToLFwGBw70vTn77gAf6gmRiDx-yL2hBDOc) and [here (115)](https://mega.nz/file/8B8nlCyI#hMzHx0KG2Ve20WqjMlFjRS6wv39Zern32eM__yQDwIw), to the demo-data/ folder. 





