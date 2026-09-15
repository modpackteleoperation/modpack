# RBY1 Robot Teleoperation (Meta Quest + Unity)

This guide shows you how to control the **RBY1 robot** using **Meta Quest** and **Unity** with real-time teleoperation. The Meta Quest sends pose and button data to a PC that translates it into robot commands via the RBY1 SDK.

---

## Requirements

- Meta Quest 2 or 3 (Developer Mode enabled)
- Unity Editor (tested on **Unity 6000.1.0f1**)
- Android Build Support (installed via Unity Hub)
- RBY1 Robot:
  - [RPC >= v0.7.0](https://github.com/RainbowRobotics/rby1-release/releases/tag/v0.7.0)
  - [SDK >= v0.7.0](https://github.com/RainbowRobotics/rby1-sdk/releases/tag/v0.7.0)

---

## Step 1: Enable Developer Mode on Meta Quest

Follow the official Meta guide here: [Meta Quest - Enable Developer Mode](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)

1. Log in to the **[Meta Developer site](https://developer.oculus.com/manage/organizations/)** and create an organization if needed.
2. On your smartphone:
   - Install the **Meta Quest mobile app**
   - Go to **Devices > Developer Mode** and enable it
3. Connect the headset to your PC via USB and accept USB debugging permission.

---

## Step 2: Unity Setup for Meta Quest

> Unity must be installed with **Android Build Support**. This setup assumes a clean project using the **Universal Render Pipeline (URP)** or **3D (Core)** template.

### 2.1 Create Unity Project

- Open Unity Hub → `New Project`
- Template: **Universal 3D**
- Project name: e.g. `RBY1_Teleoperation`

### 2.2 Import Assets

- Download and import: [MetaQuestPoseReader.unitypackage](./unity/MetaQuestPoseReader.unitypackage)
- Drag it into Unity or use `Assets > Import Package > Custom Package`

### 2.3 Add Required Packages

Go to `Window > Package Manager` and install the following:

- **XR Plug-in Management** (`com.unity.xr.management`)
- **Oculus XR Plugin** (`com.unity.xr.oculus`)
- **XR Interaction Toolkit** (`com.unity.xr.interaction.toolkit`)
- **Newtonsoft Json** (`com.unity.nuget.newtonsoft-json`)

> Tip: Use "Add package by name..." if you can't find them via search.
> - com.unity.xr.management
> - com.unity.xr.oculus
> - com.unity.xr.interaction.toolkit
> - com.unity.nuget.newtonsoft-json

Click `Window > TextMeshPro > Import TMP Essential Resources` to install **TextMesh Pro**

### 2.4 Configure XR Settings

- `Edit > Project Settings > XR Plug-in Management` → enable **Oculus** under **Android**

### 2.5 Scene Setup

- Drag `Scenes/MainScene` into **Hierarchy**
- Remove `SampleScene`

### 2.6 Build Settings

- `File > Build Profiles`
  - `Scene List`
    - Add open scenes 
    - Disable `Scenes/SampleScenes`
  - Platform: `Android` → **Switch Platform**
  - Minimum API Level: `Android 10.0 (API 29)`

### 2.7 Build and Run

- Connect Meta Quest via USB
- Click **Build and Run**
- Save as `.apk` and deploy

**Build Troubleshooting**
```
6000.2.10f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++: not found
```
Do the following: 
```
cd ~/6000.2.10f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/

mv clang clang_old
mv clang++ clang++_old
mv ld.lld ld.lld_old
mv ld64.lld ld64.lld_old
mv lld-link lld-link_old
mv llvm-strip llvm-strip_old

ln -s clang-18 clang  # Check which version of clang- is available at your bin/
ln -s clang clang++
ln -s lld ld.lld
ln -s lld ld64.lld
ln -s lld lld-link
ln -s llvm-objcopy llvm-strip
```

## Step 3: Run the Python Teleoperation Script
```bash
python main.py \
  --local_ip 192.168.***.*** \
  --meta_quest_ip 192.168.***.***
```


## Controller Mapping

| Input              | Description                        |
| ------------------ | ---------------------------------- |
| Right A Button     | Reset pose and begin teleoperation |
| Right B Button     | Stop teleoperation                 |
| Right Trigger      | Control right arm                  |
| Left Trigger       | Control left arm                   |
| Both Triggers Held | Control torso                      |

---
## Notes

- Ensure both the Meta Quest and the PC are on the **same Wi-Fi network**.
- The Meta Quest app shows its IP address on screen when running.

---
