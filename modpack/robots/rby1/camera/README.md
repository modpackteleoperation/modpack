# setting up the cameras

## If using ethernet:

- first configure the ip address - set the ethernet port's address to 192.168.88.1

## If using fiber card:

- install intel i40e driver

### Steps to Set Up the i40e Driver

**Step 1: Install Kernel Headers**

Ensure that you have the correct kernel headers installed for your current kernel.

```bash
sudo apt install linux-headers-$(uname -r)
```

This installs the necessary headers required for compiling kernel modules.

**Step 2: Download and Extract i40e Driver**

If you haven't done so already, download the driver package and extract it.

```bash
cd ~/Downloads
# Example: replace with your actual path if necessary
tar -xvf i40e-<version>.tar.gz
cd i40e-<version>/src
```

**Step 3: Build the i40e Driver**

Run the following commands to clean and build the i40e driver.

```bash
make clean
make
```

**Step 4: Sign the i40e Module (with Secure Boot)**

Since Secure Boot is enabled, you'll need to sign the i40e module so it can be loaded by the system.

1. Create Module Signing Keys

First, create a directory to store your signing keys:

```bash
sudo mkdir -p /root/module_signing_keys
```

Then generate a new private and public signing key pair:

```bash
sudo openssl req -new -x509 -newkey rsa:2048 \
  -keyout /root/module_signing_keys/MOK.priv \
  -outform DER -out /root/module_signing_keys/MOK.der \
  -nodes -days 36500 \
  -subj "/CN=Local Kernel Module Signing/"
```

This will generate:

MOK.priv (private key)

MOK.der (public key)

2. Enroll the Public Key with Secure Boot

To enable Secure Boot to trust the kernel module, enroll the public key:

```bash
sudo mokutil --import /root/module_signing_keys/MOK.der
```

Reboot your system and you’ll be prompted with the MOK Manager screen.

Select “Enroll MOK”, choose Continue, and enter the password to enroll the key.

After rebooting, you can verify that the key is enrolled using:

```bash
sudo mokutil --list-enrolled
```

It should list the key you just enrolled.

3. Sign the i40e Module

Now that the key is enrolled, sign the i40e driver module with the private key:

```bash
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
  /root/module_signing_keys/MOK.priv \
  /root/module_signing_keys/MOK.der \
  /lib/modules/$(uname -r)/updates/drivers/net/ethernet/intel/i40e/i40e.ko
```

This step ensures that Secure Boot will trust the i40e driver when loading.

**Step 5: Install the Driver**

Now, install the compiled i40e driver:

```bash
sudo make install
```

Run the depmod command to update module dependencies:

```bash
sudo depmod -a
```

**Step 6: Load the i40e Driver**

Finally, load the i40e driver into the kernel:

```bash
sudo modprobe i40e
```

**Step 7: Verify the Installation**

Verify that the i40e module is loaded:

```bash
lsmod | grep i40e
```

You should see something like:

```
i40e                  647168  0
```

**Step 8: Bring Up the Network Interface**

The interface name might be enp2s0f0np0 or something similar. Check the name with:

```bash
ip link
```

Then, bring it up:

```bash
sudo ip link set enp2s0f0np0 up
```

You can configure the IP address either manually or use Netplan/NetworkManager for a persistent configuration. Here’s how to manually assign an IP address:

```bash
sudo ip addr add 192.168.88.2/24 dev enp2s0f0np0
```

To make this persistent, use NetworkManager or Netplan:

Netplan config example:

```bash
network:
  version: 2
  renderer: networkd
  ethernets:
    enp2s0f0np0:
      dhcp4: no
      addresses:
        - 192.168.88.2/24
```

After configuring Netplan, apply changes with:

```bash
sudo netplan apply
```

**Step 9: Verify the Network Configuration**

Finally, verify the network configuration:

```bash
ip addr show enp2s0f0np0
```

You should see the configured IP address (192.168.88.2).

### install aravis

https://aravisproject.github.io/aravis/aravis-stable/building.html

```bash
sudo apt update
sudo apt upgrade
sudo apt install -f
sudo apt install libxml2-dev gobject-introspection libgirepository1.0-dev
cd ~/aravis-0.8.35
meson setup build
cd build
ninja
sudo ninja install
sudo ldconfig
```

test with 

```bash
arv-tool-0.8 list
```

install viewers

```bash
sudo apt install aravis-tools
sudo apt install gstreamer1.0-plugins-bad
```

launch viewer

```bash
arv-viewer-0.8
```

install aravis python bindings

```bash
# Make sure you're in the modpack environment
conda activate modpack

# Install PyGObject via conda
conda install -c conda-forge pygobject

# set path
export GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0:$GI_TYPELIB_PATH
```

increase max UDP packet size

```bash
ifconfig etho0 mtu 9000 up
```

or/and

```bash
vi /etc/network/interfaces
add mtu 9000
```

test camera streaming:

```bash
python3 scripts/camera_stream_viewer.py --list
```

Use one `--camera observation_key="DEVICE_ID"` flag per camera, where `DEVICE_ID`
is exactly the identifier returned by the previous command (e.g.
`FLIR-Blackfly S BFS-PGE-50S5C-XXXXXXXX`):

```bash
python3 scripts/camera_stream_viewer.py \
  --camera camera_head_main_rgb="FLIR-Blackfly S BFS-PGE-50S5C-XXXXXXXX" \
  --camera camera_left_main_rgb="FLIR-Blackfly S BFS-PGE-23S3C-XXXXXXXX"
```

When hardware is unavailable you can still validate the pipeline by generating
synthetic images:

```bash
python3 scripts/camera_stream_viewer.py --mock
```

IMPORTANT: everytime after restarting, run:

```bash
sudo /usr/local/sbin/setup-x710.sh
```

```bash
#!/bin/bash
# Configure Intel X710 port 0 with static IP

ip addr flush dev enp2s0f0 || true
ip addr add 192.168.88.2/24 dev enp2s0f0 || true
ip link set enp2s0f0 up || true
```
