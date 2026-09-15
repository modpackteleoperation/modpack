import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

def _forward_remote_stream(stream, prefix: str, log_file=None):
    """Write remote stream lines with prefix to log_file (not the terminal)."""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            if log_file is not None:
                try:
                    log_file.write(f"{prefix} {line.rstrip()}\n")
                    log_file.flush()
                except Exception:
                    pass
    except Exception as exc:
        print(f"{prefix} stream error: {exc}")
    finally:
        try:
            stream.close()
        except Exception:
            pass


class RemotePCManager:
    def __init__(
        self,
        pc_id: str,
        user: str,
        ip: str,
        workspace: str,
        manager_cfg: dict,
        scripts: List[str],
        conda_env: str = "modpack",
        arx_skip_gello_wait: bool = False,
        vp_stream_mode: str = "pcd",
        gello_pc_ip: str = "",
    ):
        self.pc_id = pc_id
        self.user = user
        self.ip = ip
        self.workspace = workspace
        self.manager_cfg = manager_cfg
        self.scripts = scripts  # commands from manifest robot_pc.scripts
        self.conda_env = conda_env
        self.arx_skip_gello_wait = arx_skip_gello_wait
        self.vp_stream_mode = vp_stream_mode
        self.gello_pc_ip = gello_pc_ip

        self.remote_processes: List[subprocess.Popen] = []
        self.log_paths: List[Path] = []
        self._log_files: List = []
        self._output_threads: List[threading.Thread] = []
        self._log_paths_by_label: dict = {}

    def launch(self) -> bool:
        """Launch robot runner scripts on the Robot PC via SSH."""
        if self.pc_id != "gello":
            print("Remote launch only available from modpack PC")
            return False

        if not self.scripts:
            print("No robot_pc.scripts defined in robot config — nothing to launch")
            return False

        print(f"\nLaunching Robot PC scripts via SSH...")
        print(f"  Target: {self.user}@{self.ip}")
        print(f"  Workspace: {self.workspace}")
        print(f"  Scripts: {self.scripts}")

        # Snapshot any leftover runners from a previous run. We kill these only
        # AFTER the new controllers are confirmed running, and by exact PID so we
        # never touch the processes we are about to start.
        stale_pids = self._remote_runner_pids()
        if stale_pids:
            print(f"  Found {len(stale_pids)} leftover Robot PC runner(s) from a "
                  f"previous run; will clean up after relaunch")

        env_prefix = ""
        if self.arx_skip_gello_wait:
            env_prefix += "ARX_SKIP_GELLO_WAIT=1 "
        if self.vp_stream_mode:
            env_prefix += f"VP_STREAM_MODE={self.vp_stream_mode} "
        if self.gello_pc_ip:
            env_prefix += f"GELLO_PC_IP={self.gello_pc_ip} "

        for script in self.scripts:
            remote_cmd = (
                f"source /home/{self.user}/miniforge3/etc/profile.d/conda.sh && "
                f"conda activate {self.conda_env} && "
                f"cd {self.workspace} && "
                f"{env_prefix}PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 {script}"
            )
            ssh_cmd = ["ssh", f"{self.user}@{self.ip}", remote_cmd]
            try:
                proc = subprocess.Popen(
                    ssh_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                label = self._script_label(script)
                log_dir = Path(tempfile.gettempdir()) / "modpack_logs"
                log_dir.mkdir(exist_ok=True)
                log_path = log_dir / f"robot_pc_{label}_{int(time.time())}.log"
                log_file = open(log_path, "w", buffering=1)
                self.remote_processes.append(proc)
                self.log_paths.append(log_path)
                self._log_files.append(log_file)
                self._log_paths_by_label[label] = log_path
                self._start_output_forwarders(proc, label, log_file)
                print(f"  Launched: {script} (SSH PID: {proc.pid}, log: {log_path})")
            except Exception as e:
                print(f"  Failed to launch '{script}': {e}")

        time.sleep(2)

        alive = []
        for proc, log_path in zip(self.remote_processes, self.log_paths):
            if proc.poll() is None:
                alive.append(proc)
            else:
                self._print_log_tail(log_path)
        if not alive:
            print("WARNING: All remote launches may have failed immediately")
            return False

        print(f"Robot PC: {len(alive)}/{len(self.remote_processes)} processes running")

        # New controllers are up — now reap the leftover runners we snapshotted
        # before launch, killing only those exact PIDs.
        if stale_pids:
            self._kill_remote_pids(stale_pids)

        return True

    def log_path_for(self, label: str) -> Optional[Path]:
        return self._log_paths_by_label.get(label)

    @staticmethod
    def _print_log_tail(log_path: Path, num_lines: int = 15):
        print(f"WARNING: Remote process exited, tail of {log_path}:")
        try:
            with open(log_path, "r") as f:
                for line in f.readlines()[-num_lines:]:
                    print(f"    {line.rstrip()}")
        except Exception as exc:
            print(f"    (failed to read log: {exc})")

    @staticmethod
    def _script_label(script: str) -> str:
        """Short unique log prefix: module name plus any word-only flags/values."""
        tokens = script.split()
        if "-m" in tokens:
            mod_idx = tokens.index("-m") + 1
            parts = [tokens[mod_idx].rsplit(".", 1)[-1]]
            parts += [t.lstrip("-") for t in tokens[mod_idx + 1:] if t.lstrip("-").isalpha()]
            return "-".join(parts[:3])
        return tokens[-1]

    def _start_output_forwarders(self, proc: subprocess.Popen, label: str, log_file=None):
        for stream, tag in [(proc.stdout, "OUT"), (proc.stderr, "ERR")]:
            if stream:
                t = threading.Thread(
                    target=_forward_remote_stream,
                    args=(stream, f"[ROBOT-{tag}|{label}]", log_file),
                    daemon=True,
                )
                t.start()
                self._output_threads.append(t)

    # Matches every Robot PC process we launch over SSH: both the runner scripts
    # (python -m modpack.robots.* / modpack.modules.*) AND the manager itself
    # (python -m modpack --pc-id robot ...), which has no dotted submodule and so
    # was previously missed — leaving orphaned managers holding sockets and stale
    # activation-bus state across restarts.
    _RUNNER_PATTERN = r"python[0-9.]* -m modpack( |\.)"

    def _remote_runner_pids(self) -> List[str]:
        """Return PIDs of robot runner processes currently live on the Robot PC."""
        pattern = self._RUNNER_PATTERN
        try:
            result = subprocess.run(
                [
                    "ssh",
                    f"{self.user}@{self.ip}",
                    f'ps -eo pid,cmd | grep -E "{pattern}" | grep -v grep',
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception as exc:
            print(f"  Warning: failed to query Robot PC runner PIDs: {exc}")
            return []

        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid = line.split(None, 1)[0]
            if pid.isdigit():
                pids.append(pid)
        return pids

    def _kill_remote_pids(self, pids: List[str]) -> None:
        """SIGKILL the given PIDs on the Robot PC."""
        if not pids:
            return
        print(f"  Cleaning up {len(pids)} leftover Robot PC runner(s): {', '.join(pids)}")
        try:
            subprocess.run(
                ["ssh", f"{self.user}@{self.ip}", f"kill -9 {' '.join(pids)}"],
                timeout=5,
                check=False,
            )
        except Exception as exc:
            print(f"  Warning: failed to kill leftover Robot PC runners: {exc}")

    def subsystems_running(self) -> bool:
        """Check via SSH if any robot runner scripts are still live."""
        pattern = self._RUNNER_PATTERN
        try:
            result = subprocess.run(
                [
                    "ssh",
                    f"{self.user}@{self.ip}",
                    f'ps -eo pid,cmd | grep -E "{pattern}" | grep -v grep',
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            output = result.stdout.strip()
            if output:
                print("  Robot PC processes still running:")
                for line in output.splitlines():
                    print(f"    {line}")
                return True
            return False
        except Exception as exc:
            print(f"  Warning: failed to query Robot PC process state: {exc}")
            return True

    def shutdown(self, publish_shutdown_fn):
        """Send shutdown signal to Robot PC and wait for it to finish."""
        if self.pc_id != "gello":
            return

        print("\nShutting down Robot PC...")

        publish_shutdown_fn()

        # Give processes a moment to receive shutdown and suspend via SIGSTOP
        time.sleep(2)
        print("Ensuring Robot PC runners exit...")
        try:
            subprocess.run(
                [
                    "ssh",
                    f"{self.user}@{self.ip}",
                    f'pkill -9 -f -- "{self._RUNNER_PATTERN}"',
                ],
                timeout=5,
                check=False,
            )
        except Exception as exc:
            print(f"  Warning: failed to signal remote runner shutdown: {exc}")

        print("Waiting for Robot PC processes to finish...")
        max_wait = 10
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if not self.subsystems_running():
                print("Robot PC processes stopped")
                break
            time.sleep(0.5)
        else:
            print(f"\nWARNING: Robot PC shutdown timeout after {max_wait}s")

        for proc in self.remote_processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()

        for f in self._log_files:
            try:
                f.close()
            except Exception:
                pass

        print("Robot PC shutdown complete")
