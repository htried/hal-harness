import asyncio
import json
import logging
import os
import shlex
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from dotenv import dotenv_values
from rich.progress import Progress, TaskID

from ..benchmarks.base_benchmark import BaseBenchmark

# Get logger for verbose output
verbose_logger = logging.getLogger('agent_eval.verbose')

# Define the docker image names
DOCKER_IMAGE_NAME = "hal-agent-runner:latest"
CRASH_TEST_IMAGE_NAME = "crash-test-hal:latest"

class DockerRunner:
    """Handles running agents in Docker containers for isolation"""
    
    def __init__(self, log_dir: str, max_concurrent: int = 1, benchmark: Optional[BaseBenchmark] = None, 
                 crash_test: bool = False, crash_test_config: Optional[Dict[str, Any]] = None):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_containers: List[str] = []
        self.benchmark = benchmark
        self.verbose = False
        self.crash_test = crash_test
        self.crash_test_config = crash_test_config or {}
        
        # Initialize Docker client
        self.docker_client = docker.from_env()
        
        # Check if Docker is available
        self._check_docker_available()
        
        # Ensure the Docker image exists
        self._ensure_docker_image()
        
    def _check_docker_available(self) -> None:
        """Check if Docker is available on the system"""
        try:
            version = self.docker_client.version()
            verbose_logger.debug(f"Docker is available: {version.get('Version', 'unknown version')}")
        except docker.errors.DockerException as e:
            error_message = "Docker is not available on this system. Please install Docker to use the Docker runner."
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
    
    def _ensure_docker_image(self) -> None:
        """Ensure the Docker image exists, building it if necessary"""
        try:
            # Choose the appropriate image based on crash_test mode
            image_name = CRASH_TEST_IMAGE_NAME if self.crash_test else DOCKER_IMAGE_NAME
            
            # Check if the image already exists
            try:
                self.docker_client.images.get(image_name)
                verbose_logger.debug(f"Docker image {image_name} already exists")
            except docker.errors.ImageNotFound:
                if self.crash_test:
                    # For crash-test mode, the image should be built externally
                    error_message = f"Crash-test image {image_name} not found. Please build it first using the build script in the crash-test-hal directory."
                    verbose_logger.debug(error_message)
                    raise RuntimeError(error_message)
                else:
                    verbose_logger.debug(f"Docker image {image_name} not found, building it...")
                    
                    # Get the Dockerfile path - it should be in the same directory as this file
                    dockerfile_dir = os.path.join(os.path.dirname(__file__), "docker")
                    dockerfile_path = os.path.join(dockerfile_dir, "Dockerfile")
                    
                    if not os.path.exists(dockerfile_path):
                        raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
                    
                    # Build the Docker image
                    verbose_logger.debug(f"Building Docker image from {dockerfile_path}")
                    
                    _, build_logs = self.docker_client.images.build(
                        path=dockerfile_dir,
                        dockerfile=os.path.basename(dockerfile_path),
                        tag=image_name
                    )
                    
                    for log in build_logs:
                        if 'stream' in log:
                            verbose_logger.debug(log['stream'].strip())
                    
                    verbose_logger.debug(f"Docker image built successfully")
                
        except docker.errors.DockerException as e:
            error_message = f"Failed to build Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            error_message = f"Error ensuring Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
        
    async def run_agent(self,
                       dataset: Dict[str, Any],
                       agent_function: str,
                       agent_dir: str,
                       agent_args: Dict[str, Any],
                       run_id: str,
                       benchmark: Optional[BaseBenchmark] = None,
                       progress: Optional[Progress] = None,
                       task: Optional[TaskID] = None,
                       timeout: int = 7200) -> Dict[str, Any]:
        """
        Run agent on all tasks with concurrency control
        """
        try:
            self.benchmark = benchmark
            # Get run directory from benchmark if provided
            run_dir = benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
            submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            
            tasks = []
            for task_id, input_data in dataset.items():
                task_coro = self._process_task(
                    task_id=task_id,
                    input_data=input_data,
                    agent_function=agent_function,
                    agent_dir=agent_dir,
                    agent_args=agent_args,
                    run_id=run_id,
                    submissions_file=submissions_file,
                    progress=progress,
                    task=task
                )
                tasks.append(task_coro)
            
            # Run tasks with concurrency control
            results = await asyncio.gather(*tasks)
            
            # Merge results
            merged_results = {}
            for result in results:
                if result:
                    merged_results.update(result)
                    
            return merged_results

        finally:
            # Cleanup any remaining containers
            for container_id in self._active_containers:
                try:
                    container = self.docker_client.containers.get(container_id)
                    # container.stop()
                    # container.remove()
                except (docker.errors.NotFound, docker.errors.APIError) as e:
                    verbose_logger.debug(f"Warning: Failed to cleanup container {container_id}: {e}")

    async def _process_task(self,
                          task_id: str,
                          input_data: Any,
                          agent_function: str,
                          agent_dir: str,
                          agent_args: Dict[str, Any],
                          run_id: str,
                          submissions_file: str,
                          progress: Optional[Progress] = None,
                          task: Optional[TaskID] = None) -> Optional[Dict[str, Any]]:
        """Process a single task with semaphore control"""
        async with self._semaphore:
            verbose_logger.debug(f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})")
            result = await self._run_single_task(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id
            )
            
            # Write result to submissions file
            if result:
                async with self._file_lock:
                    with open(submissions_file, "a") as f:
                        json.dump(result, f)
                        f.write("\n")
            
            # Update progress after task completion
            if progress and task is not None:
                progress.update(task, advance=1)
            
            verbose_logger.debug(f"Completed task {task_id}")
            return result

    async def _run_single_task(self,
                             task_id: str,
                             input_data: Any,
                             agent_function: str,
                             agent_dir: str,
                             agent_args: Dict[str, Any],
                             run_id: str,
                             timeout: int = 7200) -> Optional[Dict[str, Any]]:
        """Process a single task in a Docker container with timeout"""
        # Create temporary directory for mounting into container
        temp_dir = Path(tempfile.mkdtemp())
        container_id = f"agentrun--{uuid.uuid4()}"[:32].lower().replace("_", "-")
        
        try:
            # Copy agent code to temp directory
            temp_agent_dir = temp_dir
            shutil.copytree(agent_dir, temp_agent_dir, dirs_exist_ok=True)

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Copy task-specific files if they exist in input_data
            if isinstance(input_data, dict) and 'files' in input_data:
                for dest_path, src_path in input_data['files'].items():
                    dest_path = dest_path.replace('/root/', '').lstrip('/')
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        verbose_logger.debug(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")

            # Create runner script
            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id
            )
                        
            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)
            
            # Choose the appropriate image based on crash_test mode
            image_name = CRASH_TEST_IMAGE_NAME if self.crash_test else DOCKER_IMAGE_NAME
            
            # Prepare environment variables for crash-test mode
            container_env = {}
            if self.crash_test:
                # Add LD_PRELOAD to load libnoisy in crash-test mode
                container_env['LD_PRELOAD'] = '/utils/libnoisy.so'
                # Also add the crash-test configuration variables
                container_env['NETWORK_FAILURE_RATE'] = str(self.crash_test_config.get("failure_rate", "0.5"))
                container_env['NOISY_ERROR_MODE'] = str(self.crash_test_config.get("error_mode", "4xx_errors"))
                container_env['NOISY_MODE'] = str(self.crash_test_config.get("noisy_mode", "both"))
                # Add allowed domains
                if "allowed_domains" in self.crash_test_config:
                    container_env['ALLOWED_DOMAINS'] = ",".join(self.crash_test_config["allowed_domains"])
                else:
                    default_allowed = [
                        "api.openai.com", "api.anthropic.com", 
                        "generativelanguage.googleapis.com",
                        "bedrock-runtime.us-east-1.amazonaws.com",
                        "api.groq.com", "api.deepseek.com", "api.x.ai",
                        "api.wandb.ai", "wandb.ai", "github.com"
                    ]
                    container_env['ALLOWED_DOMAINS'] = ",".join(default_allowed)
                # Add debug flag if specified
                if self.crash_test_config.get("debug", False):
                    container_env['NOISY_DEBUG'] = "1"
            
            # Add all necessary environment variables from .env file to container
            # This ensures API keys (like WANDB_API_KEY) are available in the container
            dotenv_vars = dotenv_values(".env")
            for key, value in dotenv_vars.items():
                if value is not None:  # Only add non-None values
                    container_env[key] = value
            
            # create container from image and mount temp dir
            container = self.docker_client.containers.run(
                image=image_name,
                name=container_id,
                detach=True,
                command=["tail", "-f", "/dev/null"],  # Keep container running
                environment=container_env,
            )
            
            # Add container to active list
            self._active_containers.append(container_id)
            
            # Using asyncio subprocess instead of subprocess.run
            # copy all the contents of temp dir into container
            proc = await asyncio.create_subprocess_exec(
                "docker", "cp", f"{temp_dir}/.", f"{container_id}:/workspace",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
            
            # create env
            # IMPORTANT: Unset LD_PRELOAD for conda/pip operations - they need real network access!
            create_env_cmd = (
                "unset LD_PRELOAD && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && "
                "conda create -y -n agent_env python=3.12"
            )
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", create_env_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if self.verbose or proc.returncode != 0:
                if stdout:
                    verbose_logger.debug(f"Container {container_id} [conda create]: {stdout.decode()}")
                if stderr:
                    verbose_logger.debug(f"Container {container_id} [conda create]: {stderr.decode()}")
            
            if proc.returncode != 0:
                verbose_logger.warning(f"conda create failed with return code {proc.returncode}")
            else:
                verbose_logger.debug(f"conda environment 'agent_env' created successfully")

            # Use full path to conda in Docker containers
            conda_cmd = "/opt/conda/bin/conda"
            
            # Verify environment exists
            check_env_cmd = f"unset LD_PRELOAD && {conda_cmd} info --envs | grep agent_env"
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", check_env_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if stdout:
                verbose_logger.debug(f"Container {container_id} [env check]: {stdout.decode()}")
            if stderr and stderr.decode().strip():
                verbose_logger.debug(f"Container {container_id} [env check stderr]: {stderr.decode()}")
                
            # install requirements
            # IMPORTANT: Unset LD_PRELOAD for pip - we need to download packages
            pip_cmd = f"unset LD_PRELOAD && {conda_cmd} run --no-capture-output -n agent_env pip install -r /workspace/requirements.txt"
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", pip_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
            
            # Get current environment variables
            # Note: For crash-test mode, environment variables are already set in container_env
            # so they're inherited by the container and don't need to be passed again
            env_vars = os.environ.copy()
                        
            # run setup script if it exists
            if self.benchmark and self.benchmark.setup_script:
                print(f"Running setup script: {self.benchmark.setup_script}")
                setup_script_src = Path(self.benchmark.setup_script)
                if setup_script_src.exists():
                    # copy setup script to container
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "cp", f"{setup_script_src}", f"{container_id}:/workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:
                        if stdout:
                            verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
                    
                    # run setup script and wait for it to complete
                    # IMPORTANT: Unset LD_PRELOAD during setup too!
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "exec", container_id, "bash", "-c", "unset LD_PRELOAD && bash /workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:    
                        if stdout:
                            verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")   
                        
            # install weave
            # IMPORTANT: Unset LD_PRELOAD for weave install too!
            weave_cmd = f"unset LD_PRELOAD && {conda_cmd} run --no-capture-output -n agent_env pip install weave==0.51.41 'gql<4'"
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", weave_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")                    
            
            # Run the script and capture output with timeout handling
            start_time = time.time() 
        
            # get env vars from .env file and merge with crash-test vars
            dotenv_vars = dotenv_values(".env")
            env_vars.update(dotenv_vars)

            # Filter out problematic environment variables that cause issues in Docker
            problematic_vars = [
                'CONDA_PROMPT_MODIFIER', 'CONDA_SHLVL', 'CONDA_PREFIX_1', 
                'VSCODE_GIT_ASKPASS_NODE', 'VSCODE_DEBUGPY_ADAPTER_ENDPOINTS', 
                'BUNDLED_DEBUGPY_PATH', 'CONDA_EXE', 'CONDA_PYTHON_EXE', 
                'CONDA_DEFAULT_ENV', 'CONDA_PREFIX',
                # Filter out variables that point to host paths that don't exist in container
                'TMPDIR', 'HOME', 'USER', 'LOGNAME',
                # Filter out terminal and session variables
                'TERM', 'TERM_PROGRAM', 'TERM_PROGRAM_VERSION',
                # Filter out Cursor/VSCode specific variables
                'CURSOR_TRACE_ID', 'VSCODE_GIT_IPC_HANDLE',
                # Filter out local paths
                'PWD', 'PATH', 'SHELL', 'NVM_DIR', 'NVM_INC', 'NVM_BIN',
                'COMMAND_MODE', '__CF_USER_TEXT_ENCODING',
                'TIKTOKEN_CACHE_DIR', 'PYDEVD_DISABLE_FILE_VALIDATION'
            ]
            filtered_env_vars = {k: v for k, v in env_vars.items() if k not in problematic_vars}

            # Properly quote environment variable values to handle special characters
            env_vars_str = " ".join([f"{k}={shlex.quote(str(v))}" for k, v in filtered_env_vars.items()])
            print(f"Running script with env: {env_vars_str}")
            
            # If in crash-test mode, we need to pass LD_PRELOAD to conda run
            # The issue is that conda run doesn't pass LD_PRELOAD to the Python subprocess
            # So we need to explicitly set it in the bash environment before conda run
            if self.crash_test:
                # Export all environment variables in the bash session
                # This ensures they're available when Python starts via conda run
                export_cmds = " && ".join([
                    "export LD_PRELOAD=/utils/libnoisy.so",
                    f"export NETWORK_FAILURE_RATE={self.crash_test_config.get('failure_rate', '1.0')}",
                    f"export NOISY_ERROR_MODE={self.crash_test_config.get('error_mode', '4xx_errors')}",
                    f"export NOISY_MODE={self.crash_test_config.get('noisy_mode', 'both')}",
                    f"export ALLOWED_DOMAINS='{container_env.get('ALLOWED_DOMAINS', '')}'"
                ])
                bash_cmd = f"{export_cmds} && echo '[CRASH-TEST DEBUG] Environment variables:' && env | grep -E '(LD_PRELOAD|NETWORK_FAILURE_RATE|NOISY_ERROR_MODE|NOISY_MODE|ALLOWED_DOMAINS)' && {conda_cmd} run --no-capture-output -n agent_env python run_agent.py"
            else:
                bash_cmd = f"{env_vars_str} {conda_cmd} run -n agent_env python run_agent.py"

            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id, "bash", "-c", bash_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if stdout:
                verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")        
            
            # Poll for output.json with timeout
            result = None
            while time.time() - start_time < timeout:
                # Check if output.json exists
                check_result = container.exec_run(["test", "-f", "/workspace/output.json"])
                if check_result.exit_code == 0:
                    # copy files from container back to host
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "cp", f"{container_id}:/workspace/.", f"{temp_dir}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()                    
                    if stdout:
                        verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
                    
                    # Load and return results
                    with open(temp_dir / "output.json") as f:
                        result = json.load(f)
                        break
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            if result is None:
                verbose_logger.debug(f"Task {task_id} timed out after {timeout} seconds")
                return {task_id: f"TIMEOUT after {timeout} seconds"}
            
            return result

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            # Cleanup
            try:
                # Copy directory to log_dir if specified
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    shutil.copytree(temp_dir, task_log_dir, dirs_exist_ok=True)
                
                # Remove temp directory
                shutil.rmtree(temp_dir)
                
                # Remove container
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.remove(force=True)
                    # Remove from active containers list
                    if container_id in self._active_containers:
                        self._active_containers.remove(container_id)
                except Exception:
                    pass  # Container may already be removed
                
            except Exception as e:
                error_msg = f"Warning: Failed to cleanup for task {task_id}: {e}"
                verbose_logger.debug(error_msg)

    def _create_runner_script(self, agent_function: str, task_id: str, run_id: str) -> str:
        """
        Create the Python script that will run the agent
        """
        module_name, function_name = agent_function.rsplit(".", 1)
        
        return f'''
import os
import json
import importlib.util
import traceback
import weave

try:
    # Debug: Print environment variables for crash-test mode (always check, useful for debugging)
    if os.environ.get('NETWORK_FAILURE_RATE'):
        print("=" * 60)
        print("CRASH-TEST DEBUG: Environment variables in Python")
        print(f"NETWORK_FAILURE_RATE: {{os.environ.get('NETWORK_FAILURE_RATE', 'NOT SET')}}")
        print(f"NOISY_ERROR_MODE: {{os.environ.get('NOISY_ERROR_MODE', 'NOT SET')}}")
        print(f"NOISY_MODE: {{os.environ.get('NOISY_MODE', 'NOT SET')}}")
        print(f"ALLOWED_DOMAINS: {{os.environ.get('ALLOWED_DOMAINS', 'NOT SET')}}")
        print(f"LD_PRELOAD: {{os.environ.get('LD_PRELOAD', 'NOT SET')}}")
        print("=" * 60)
    
    # Initialize weave
    weave.init("{run_id}")
    
    # Load input data
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join(os.getcwd(), "{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "{function_name}")
    
    # Run the agent function
    with weave.attributes({{"weave_task_id": "{task_id}"}}):
        result = agent_fn(input_data, **agent_args)
    
    # Save output
    with open("output.json", "w") as f:
        json.dump(result, f)

except Exception as e:
    print(f"Error running agent: {{e}}")
    print(traceback.format_exc())
    with open("error.log", "w") as f:
        f.write(f"ERROR: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    raise
''' 
