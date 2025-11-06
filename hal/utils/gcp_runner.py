"""
Google Cloud Compute Engine runner for parallel benchmark execution.

This module provides a runner that executes agents on Google Cloud VMs
in parallel, similar to the Azure VMRunner but using GCP Compute Engine.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
from typing import Any, Dict, Optional

from rich.progress import Progress, TaskID

from ..benchmarks.base_benchmark import BaseBenchmark
from .gcp_utils import VirtualMachineManager


class GCPRunner:
    """Handles running agents on Google Cloud VMs"""
    
    def __init__(self, log_dir: str, max_concurrent: int = 1, benchmark: Optional[BaseBenchmark] = None):
        self.max_concurrent = max_concurrent
        self.log_dir = log_dir
        self.vm_manager = VirtualMachineManager()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_vms: list[str] = []
        self.benchmark = benchmark
        
    async def fetch_agent_logs(self, vm_name, username, ssh_private_key_path, task_id):
        """Fetch the latest agent trace log from a VM and store it locally."""
        try:
            # For now, return empty - could implement log fetching similar to Azure
            pass
        except Exception as e:
            print(f"Error fetching logs for {task_id}: {e}")
    
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
        """Run agent on all tasks using Google Cloud VMs"""
        self.benchmark = benchmark
        results = {}
        vm_names = []
        
        async def process_task(task_id: str, input_data: Any) -> Optional[Dict]:
            # Create unique VM name with timestamp and random suffix
            timestamp = int(time.time())
            random_suffix = uuid.uuid4().hex[:6]
            vm_name = f"hal-{benchmark.benchmark_name[:8]}-{timestamp}-{random_suffix}"[:62].lower().replace("_", "-")
            vm_names.append(vm_name)
            
            try:
                # Create VM
                print(f"Creating VM {vm_name} for task {task_id}")
                vm_info = await asyncio.to_thread(
                    self.vm_manager.create_vm,
                    vm_name=vm_name,
                    username="ubuntu",
                    ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                    machine_type="n1-standard-4"
                )
                
                # Wait for VM to be ready and Docker to be installed
                print(f"Waiting for VM {vm_name} to be ready...")
                await asyncio.sleep(30)  # Give startup script time to run
                
                # Create temp directory with all necessary files
                temp_dir = tempfile.mkdtemp()
                try:
                    # Create input and args files
                    input_file = os.path.join(temp_dir, 'input.json')
                    args_file = os.path.join(temp_dir, 'agent_args.json')
                    
                    with open(input_file, 'w') as f:
                        json.dump({task_id: input_data}, f)
                    with open(args_file, 'w') as f:
                        json.dump(agent_args, f)

                    # Copy task-specific files if they exist in input_data
                    if isinstance(input_data, dict) and 'files' in input_data:
                        for dest_path, src_path in input_data['files'].items():
                            dest_path = dest_path.replace('/root/', '').lstrip('/')
                            dest_full_path = os.path.join(temp_dir, dest_path)
                            os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)
                            
                            try:
                                if os.path.isdir(src_path):
                                    shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src_path, dest_full_path)
                            except Exception as e:
                                print(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")

                    # Copy setup script if it exists
                    if self.benchmark and self.benchmark.setup_script:
                        setup_script_src = os.path.join(self.benchmark.setup_script)
                        if os.path.exists(setup_script_src):
                            setup_script_dest = os.path.join(temp_dir, 'setup_script.sh')
                            shutil.copy2(setup_script_src, setup_script_dest)
                            os.chmod(setup_script_dest, 0o755)

                    # Copy files to VM
                    print(f"Copying files to VM {vm_name}")
                    await asyncio.to_thread(
                        self.vm_manager.copy_files_to_vm,
                        source_directory=temp_dir,
                        vm_name=vm_name,
                        username="ubuntu", 
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                    )
                    
                    # Copy agent directory to VM
                    await asyncio.to_thread(
                        self.vm_manager.copy_files_to_vm,
                        source_directory=agent_dir,
                        vm_name=vm_name,
                        username="ubuntu",
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        remote_path="/tmp/hal-agent"
                    )

                    # Run agent on VM
                    print(f"Running agent on VM {vm_name} for task {task_id}")
                    await asyncio.to_thread(
                        self.run_agent_on_vm,
                        vm_name=vm_name,
                        agent_function=agent_function,
                        task_id=task_id,
                        input_data=input_data,
                        agent_args=agent_args,
                        agent_dir="/tmp/hal-agent",
                        run_id=run_id,
                        username="ubuntu",
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        log_dir=self.log_dir,
                        benchmark=benchmark,
                        timeout=timeout
                    )
                    
                    # Wait for completion and get results
                    print(f"Waiting for task {task_id} to complete...")
                    result = await asyncio.to_thread(
                        self.check_task_completion,
                        vm_name=vm_name,
                        username="ubuntu",
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        timeout=timeout
                    )
                    
                    if result is None:
                        return {task_id: f"TIMEOUT after {timeout} seconds"}

                    # Copy results back
                    if self.log_dir:
                        print(f"Copying results from VM {vm_name}")
                        dest_dir = os.path.join(self.log_dir, task_id)
                        os.makedirs(dest_dir, exist_ok=True)
                        await asyncio.to_thread(
                            self.vm_manager.copy_files_from_vm,
                            vm_name=vm_name,
                            username="ubuntu",
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                            remote_path="/tmp/hal/output",
                            destination_directory=dest_dir
                        )

                    return result

                finally:
                    shutil.rmtree(temp_dir)

            except Exception as e:
                print(f"Error processing task {task_id} on VM {vm_name}: {e}")
                traceback.print_exc()
                return {task_id: f"ERROR: {str(e)}"}
            
            finally:
                # Cleanup VM
                try:
                    print(f"Deleting VM {vm_name}")
                    await asyncio.to_thread(self.vm_manager.delete_vm, vm_name)
                    if progress and task is not None:
                        progress.update(task, advance=1)
                except Exception as e:
                    print(f"Error deleting VM {vm_name}: {e}")

        # Run tasks in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_semaphore(task_id, input_data):
            async with semaphore:
                return await process_task(task_id, input_data)

        # Create tasks for all inputs
        tasks = [run_with_semaphore(task_id, input_data) 
                 for task_id, input_data in dataset.items()]
        
        # Run all tasks and gather results
        results = await asyncio.gather(*tasks)
        
        # Merge results
        merged_results = {}
        for result in results:
            if result:
                merged_results.update(result)

        # Save raw submissions if log_dir provided
        if self.log_dir:
            raw_submissions_path = os.path.join(self.log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            os.makedirs(self.log_dir, exist_ok=True)
            
            # append to submissions file
            with open(raw_submissions_path, "a") as f:
                for task_id, result in merged_results.items():
                    json.dump({task_id: result}, f)
                    f.write('\n')

        return merged_results
    
    def run_agent_on_vm(self, agent_function, vm_name, task_id, input_data, agent_args, agent_dir, 
                       run_id, username, log_dir, ssh_private_key_path, benchmark, timeout=7200):
        """Execute the agent on a VM using Docker"""
        # Create runner script
        module_name, function_name = agent_function.rsplit(".", 1)
        
        runner_script = f'''#!/bin/bash
set -e

# Pull Docker image
docker pull us-central1-docker.pkg.dev/{os.getenv("GCP_PROJECT_ID")}/hal-artifacts/hal-agent-runner:latest || \
docker pull hal-agent-runner:latest || true

# Set up environment
cd /tmp/hal
export PYTHONUNBUFFERED=1

# Create Docker container and run agent
docker run --rm \\
  -v $(pwd):/workspace \\
  -w /workspace \\
  -e TASK_ID={task_id} \\
  -e RUN_ID={run_id} \\
  -e AGENT_FUNCTION={agent_function} \\
  -e AGENT_DIR={agent_dir} \\
  --env-file <(env | grep -E '^OPENAI_API_KEY|^ANTHROPIC_API_KEY') \\
  hal-agent-runner:latest \\
  bash -c "
    python << 'PYTHON_SCRIPT'
import os
import json
import importlib.util
import traceback

try:
    # Load input and args
    with open('input.json', 'r') as f:
        input_data = json.load(f)
    with open('agent_args.json', 'r') as f:
        agent_args = json.load(f)
    
    # Import and run agent
    spec = importlib.util.spec_from_file_location(
        '{module_name}',
        '{module_name}.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, '{function_name}')
    
    result = agent_fn(input_data, **agent_args)
    
    # Save output
    os.makedirs('output', exist_ok=True)
    with open('output/output.json', 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    print(f'ERROR: {{e}}')
    traceback.print_exc()
    os.makedirs('output', exist_ok=True)
    with open('output/error.log', 'w') as f:
        f.write(f'ERROR: {{str(e)}}\\n')
        f.write(traceback.format_exc())
    raise
PYTHON_SCRIPT
  "
'''
        
        # Write and execute script
        script_path = "/tmp/run_agent.sh"
        return_code = self.vm_manager.execute_command_on_vm(
            vm_name=vm_name,
            username=username,
            ssh_private_key_path=ssh_private_key_path,
            command=f"cat > {script_path} << 'SCRIPT_END'\n{runner_script}\nSCRIPT_END\nchmod +x {script_path}",
            timeout=60
        )[2]
        
        if return_code != 0:
            raise RuntimeError(f"Failed to create runner script on VM")
        
        # Execute the script
        stdout, stderr, return_code = self.vm_manager.execute_command_on_vm(
            vm_name=vm_name,
            username=username,
            ssh_private_key_path=ssh_private_key_path,
            command=f"bash {script_path}",
            timeout=timeout
        )
        
        if return_code != 0:
            print(f"Error running agent on VM: {stderr}")
    
    def check_task_completion(self, vm_name, username, ssh_private_key_path, timeout=7200):
        """Check if a task has completed and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if output.json exists
            check_cmd = "test -f /tmp/hal/output/output.json && echo 'COMPLETE' || echo 'INCOMPLETE'"
            stdout, stderr, return_code = self.vm_manager.execute_command_on_vm(
                vm_name=vm_name,
                username=username,
                ssh_private_key_path=ssh_private_key_path,
                command=check_cmd,
                timeout=30
            )
            
            if stdout.strip() == "COMPLETE":
                # Get the result
                get_result_cmd = "cat /tmp/hal/output/output.json"
                stdout, stderr, return_code = self.vm_manager.execute_command_on_vm(
                    vm_name=vm_name,
                    username=username,
                    ssh_private_key_path=ssh_private_key_path,
                    command=get_result_cmd,
                    timeout=30
                )
                
                if stdout:
                    try:
                        return json.loads(stdout)
                    except json.JSONDecodeError:
                        pass
            
            time.sleep(10)  # Check every 10 seconds
        
        return None  # Timeout


