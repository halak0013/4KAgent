import os
import subprocess
import time
from pathlib import Path
from typing import Optional


class Tool:
    """Abstract class for a tool.

    Args:
        tool_name (str): Tool name, a valid identifier serving as the name of environment, configuration file, etc.
        subtask (str): Subtask name, serving as the name of the directory for the subtask.
        work_dir (str | None, optional): Basename of working directory. Defaults to None.
        script_rel_path (Path | str | None, optional): Path relative to the working directory of the script to run. Defaults to None.
    """

    def __init__(
        self,
        tool_name: str,
        subtask: str,
        work_dir: Optional[Path] = None,
        script_rel_path: Optional[Path | str] = None
    ):
        self.tool_name = tool_name
        self.subtask = subtask
        self.work_dir: Optional[Path] = None
        self.script_path: Optional[Path] = None
        if work_dir is not None:
            assert script_rel_path is not None, "If `work_dir` is provided, `script_rel_path` should also be provided."
            self.work_dir = Path().resolve() / 'executor' / subtask / 'tools' / work_dir
            self.script_path = self.work_dir / script_rel_path

    def __call__(self, input_dir: Path, output_dir: Path, silent: bool = False, run_gpu_id: Optional[int] = None, *args) -> None:
        if not silent:
            print('-' * 100)
            print(f"Subtask\t: {self.subtask}")
            print(f"Tool\t: {self.tool_name}")
            print(f"Input\t: {list(input_dir.glob('*'))[0]}")

        start_time = time.time()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.run_gpu_id = run_gpu_id

        self._precheck()
        self._invoke(*args)
        self._postcheck()

        end_time = time.time()
        if not silent:
            print(f"Output\t: {list(output_dir.glob('*'))[0]}")
            print(f"Time\t: {round(end_time - start_time, 3)}s")

    def _precheck(self) -> None:
        assert len(os.listdir(self.input_dir)) == 1, "The input directory should contain the input only."
        assert os.listdir(self.output_dir) == [], "The output directory should be empty."

    def _postcheck(self) -> None:
        output = [file for file in self.output_dir.glob('*') if file.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
        
        if len(output) == 0:
            # Tool didn't produce any output - provide detailed error message
            all_files = list(self.output_dir.glob('*'))
            error_msg = f"Tool '{self.tool_name}' produced no output in {self.output_dir}."
            if all_files:
                error_msg += f" Found {len(all_files)} file(s): {[f.name for f in all_files]}"
            raise AssertionError(error_msg)
        
        # If multiple image files exist, keep only the first one (by modification time)
        if len(output) > 1:
            output = sorted(output, key=lambda x: x.stat().st_mtime)
            output_file = output[0]
            # Remove other image files
            for extra_file in output[1:]:
                extra_file.unlink()
        else:
            output_file = output[0]
        
        # Rename to output.png if not already named that
        if output_file.name != 'output.png':
            output_file.replace(self.output_dir / 'output.png')
        
        # Clean up any non-image files
        for file in self.output_dir.glob('*'):
            if file.is_file() and file.suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                file.unlink()

    def _invoke(self, *args) -> None:
        self._preprocess()
        
        # tools need specific conda environment
        if self.tool_name in {
            "fourierdiff", 
            "autodir", "diffplugin", 
            "maxim", 
            'nafnet', 'nafnet_2x',
            "EVSSM", 'ridcp', "AdaRevD", "FFTformer", "MLWNet", "UFPDeblur", "Turtle",
            "osediff", "osediff_2x", "osediff_16x", 
            "pisasr", "pisasr_2x", "pisasr_16x", "pisasr_psnr", "pisasr_2x_psnr"
        }:
            cmd = self._get_cmd_with_envs()
        else:
            cmd = self._get_cmd()

        try:
            subprocess.run(cmd, cwd=self.work_dir, shell=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            # Re-run to capture output for debugging
            result = subprocess.run(cmd, cwd=self.work_dir, shell=True, check=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            error_output = result.stdout if result.stdout else "(no output captured)"
            raise RuntimeError(f"Tool '{self.tool_name}' failed with exit code {result.returncode}.\nError output:\n{error_output}")
        
        self._postprocess()

    def _get_cmd(self) -> str:
        opts = self._get_cmd_opts()
        if self.run_gpu_id is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={self.run_gpu_id} python '{self.script_path}'"
        else:
            cmd = f"python '{self.script_path}'"
        for opt in opts:
            cmd += f" '{opt}'"
        return cmd
    
    def _get_cmd_with_envs(self) -> str:
        opts = self._get_cmd_opts()

        if self.tool_name == "ridcp":
            env_name = "4kagent_dehaze"
        elif self.tool_name in ["osediff", "osediff_2x", "osediff_16x", "pisasr", "pisasr_2x", "pisasr_16x", "pisasr_psnr", "pisasr_2x_psnr"]:
            env_name = "4kagent_sr"
        else:
            env_name = "4kagent_spec_tools"
        
        if self.run_gpu_id is not None:
            cmd = f"CUDA_VISIBLE_DEVICES={self.run_gpu_id} conda run -n {env_name} python '{self.script_path}'"
        else:
            cmd = f"conda run -n {env_name} python '{self.script_path}'"

        for opt in opts:
            cmd += f" '{opt}'"
        return cmd

    def _get_cmd_opts(self, *args) -> list[str]:
        raise NotImplementedError

    def _preprocess(self) -> None:
        pass

    def _postprocess(self) -> None:
        pass
