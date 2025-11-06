"""
Google Cloud Platform utilities for managing Compute Engine VMs.

This module provides functionality to create, manage, and interact with
Google Cloud VMs for parallel benchmark execution.
"""

import asyncio
import json
import os
import tarfile
import tempfile
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Dict, Optional

import paramiko
from google.auth import default as auth_default
from google.cloud import compute_v1
from tenacity import retry, stop_after_attempt, wait_exponential


# Define retry decorator with tenacity
def get_retry_decorator(max_attempts=3, initial_wait=1, max_wait=30):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        reraise=True
    )


class VirtualMachineManager:
    """Manages Google Cloud Compute Engine VMs"""
    
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.zone = os.getenv("GCP_ZONE", "us-central1-a")
        self.region = self.zone.rsplit("-", 1)[0]
        
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable must be set")
        
        # Initialize credentials and clients
        self.credentials, _ = auth_default()
        self.instances_client = compute_v1.InstancesClient()
        self.zone_ops_client = compute_v1.ZoneOperationsClient()
        self.images_client = compute_v1.ImagesClient()
        
    def _wait_for_operation(self, operation_name: str, timeout: int = 600):
        """Wait for a compute operation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            operation = self.zone_ops_client.get(
                project=self.project_id,
                zone=self.zone,
                operation=operation_name
            )
            
            if operation.status == "DONE":
                if operation.error:
                    raise RuntimeError(f"Operation failed: {operation.error}")
                return
            
            time.sleep(2)
        
        raise TimeoutError(f"Operation {operation_name} timed out")
    
    @get_retry_decorator()
    def create_vm(self, vm_name: str, username: str = "agent", ssh_public_key_path: Optional[str] = None, 
                  machine_type: str = "n1-standard-4", disk_size_gb: int = 100) -> Dict:
        """
        Create a Google Cloud VM instance.
        
        Args:
            vm_name: Unique name for the VM
            username: Username for SSH access
            ssh_public_key_path: Path to SSH public key file
            machine_type: Machine type (e.g., n1-standard-4)
            disk_size_gb: Boot disk size in GB
            
        Returns:
            Dict with VM information including IP addresses
        """
        print(f"Creating VM {vm_name} in zone {self.zone}...")
        
        # Read SSH public key
        ssh_public_key = None
        if ssh_public_key_path:
            # Expand ~ in path
            expanded_path = os.path.expanduser(ssh_public_key_path)
            with open(expanded_path, "r") as f:
                ssh_public_key = f.read().strip()
        
        # Get the latest Ubuntu image
        image_response = self.images_client.get_from_family(
            project="ubuntu-os-cloud",
            family="ubuntu-2204-lts"
        )
        
        # Configure the VM
        metadata_items = []
        
        # Startup script to install Docker
        artifact_registry = os.getenv("ARTIFACT_REGISTRY", "hal-artifacts")
        region = os.getenv("GCP_REGION", "us-central1")
        
        startup_script = f"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# Update system
apt-get update
apt-get install -y docker.io docker-compose git python3-pip

# Configure Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Install Google Cloud CLI for artifact registry access
echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y google-cloud-cli

# Configure Docker to use Artifact Registry
region="{region}"
registry_url="${{region}}-docker.pkg.dev"
gcloud auth configure-docker ${{registry_url}} --quiet || true

echo "VM setup complete at $(date)"
"""
        
        metadata_items.append(
            compute_v1.Items(key="startup-script", value=startup_script)
        )
        
        # Add SSH key to metadata if provided
        if ssh_public_key:
            existing_keys = f"{username}:{ssh_public_key}"
            metadata_items.append(
                compute_v1.Items(key=f"ssh-keys", value=existing_keys)
            )
        
        instance = compute_v1.Instance(
            name=vm_name,
            machine_type=f"zones/{self.zone}/machineTypes/{machine_type}",
            disks=[
                compute_v1.AttachedDisk(
                    boot=True,
                    auto_delete=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        disk_size_gb=disk_size_gb,
                        source_image=image_response.self_link,
                    ),
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    access_configs=[
                        compute_v1.AccessConfig(
                            name="External NAT",
                            type_="ONE_TO_ONE_NAT"
                        )
                    ],
                    network="global/networks/default"
                )
            ],
            service_accounts=[
                compute_v1.ServiceAccount(
                    email="default",
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            ],
            metadata=compute_v1.Metadata(items=metadata_items),
            tags=compute_v1.Tags(items=["hal-agent-vm"]),
            labels={"purpose": "hal-benchmark", "created-by": "hal-harness"}
        )
        
        # Create the VM
        operation = self.instances_client.insert(
            project=self.project_id,
            zone=self.zone,
            instance_resource=instance
        )
        
        # Wait for the operation to complete
        self._wait_for_operation(operation.name)
        
        # Get the created VM
        vm_info = self.instances_client.get(
            project=self.project_id,
            zone=self.zone,
            instance=vm_name
        )
        
        # Get IP addresses
        internal_ip = vm_info.network_interfaces[0].network_i_p
        external_ip = vm_info.network_interfaces[0].access_configs[0].nat_i_p
        
        return {
            "name": vm_name,
            "internal_ip": internal_ip,
            "external_ip": external_ip,
            "status": vm_info.status,
            "self_link": vm_info.self_link
        }
    
    @get_retry_decorator()
    def get_vm_ip(self, vm_name: str) -> Dict[str, str]:
        """Get the IP addresses of a VM"""
        vm_info = self.instances_client.get(
            project=self.project_id,
            zone=self.zone,
            instance=vm_name
        )
        
        internal_ip = vm_info.network_interfaces[0].network_i_p
        external_ip = vm_info.network_interfaces[0].access_configs[0].nat_i_p
        
        return {
            "internal_ip": internal_ip,
            "external_ip": external_ip
        }
    
    @get_retry_decorator()
    def delete_vm(self, vm_name: str):
        """Delete a VM instance"""
        print(f"Deleting VM {vm_name}...")
        operation = self.instances_client.delete(
            project=self.project_id,
            zone=self.zone,
            instance=vm_name
        )
        self._wait_for_operation(operation.name)
    
    def copy_files_to_vm(self, source_directory: str, vm_name: str, username: str, 
                         ssh_private_key_path: str, remote_path: str = "/tmp/hal"):
        """
        Copy files to VM using SCP.
        
        Args:
            source_directory: Local directory to copy
            vm_name: Name of the VM
            username: SSH username
            ssh_private_key_path: Path to SSH private key
            remote_path: Remote directory path on VM
        """
        # Create a tarball of the source directory
        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp_tar:
            tar_path = tmp_tar.name
            
            with tarfile.open(tar_path, 'w') as tar:
                tar.add(source_directory, arcname='.')
            
            try:
                # Get VM external IP
                vm_info = self.instances_client.get(
                    project=self.project_id,
                    zone=self.zone,
                    instance=vm_name
                )
                external_ip = vm_info.network_interfaces[0].access_configs[0].nat_i_p
                
                # Copy tar file to VM
                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                expanded_key_path = os.path.expanduser(ssh_private_key_path)
                ssh_key = paramiko.RSAKey.from_private_key_file(expanded_key_path)
                
                ssh_client.connect(
                    hostname=external_ip,
                    username=username,
                    pkey=ssh_key,
                    timeout=60
                )
                
                # Create remote directory
                sftp = ssh_client.open_sftp()
                
                # Copy tar file
                sftp.put(tar_path, f"/tmp/hal_files.tar")
                sftp.close()
                
                # Extract tar on remote host
                command = f"mkdir -p {remote_path} && cd {remote_path} && tar -xf /tmp/hal_files.tar && rm /tmp/hal_files.tar"
                stdin, stdout, stderr = ssh_client.exec_command(command, timeout=300)
                
                # Wait for command to complete
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    error_output = stderr.read().decode()
                    raise RuntimeError(f"Failed to extract files on VM: {error_output}")
                
                ssh_client.close()
                
            finally:
                # Clean up local tar file
                os.unlink(tar_path)
    
    def copy_files_from_vm(self, vm_name: str, username: str, ssh_private_key_path: str, 
                          remote_path: str, destination_directory: str):
        """Copy files from VM using SCP"""
        # Get VM external IP
        vm_info = self.instances_client.get(
            project=self.project_id,
            zone=self.zone,
            instance=vm_name
        )
        external_ip = vm_info.network_interfaces[0].access_configs[0].nat_i_p
        
        # SSH into VM and create tar
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_key = paramiko.RSAKey.from_private_key_file(ssh_private_key_path)
        
        ssh_client.connect(
            hostname=external_ip,
            username=username,
            pkey=ssh_key,
            timeout=60
        )
        
        # Create tar on remote host
        remote_tar_path = f"/tmp/hal_results_{int(time.time())}.tar"
        command = f"cd {remote_path} && tar -cf {remote_tar_path} ."
        stdin, stdout, stderr = ssh_client.exec_command(command, timeout=300)
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status != 0:
            error_output = stderr.read().decode()
            raise RuntimeError(f"Failed to create tar on VM: {error_output}")
        
        # Copy tar from VM
        sftp = ssh_client.open_sftp()
        local_tar_path = os.path.join(destination_directory, "results.tar")
        os.makedirs(destination_directory, exist_ok=True)
        
        sftp.get(remote_tar_path, local_tar_path)
        sftp.close()
        
        # Extract locally
        with tarfile.open(local_tar_path, 'r') as tar:
            tar.extractall(destination_directory)
        
        # Clean up
        os.unlink(local_tar_path)
        
        # Clean up remote tar
        stdin, stdout, stderr = ssh_client.exec_command(f"rm {remote_tar_path}", timeout=10)
        ssh_client.close()
    
    def execute_command_on_vm(self, vm_name: str, username: str, ssh_private_key_path: str, 
                              command: str, timeout: int = 300) -> tuple[str, str, int]:
        """
        Execute a command on the VM and return stdout, stderr, and exit code.
        
        Returns:
            (stdout, stderr, exit_status) tuple
        """
        # Get VM external IP
        vm_info = self.instances_client.get(
            project=self.project_id,
            zone=self.zone,
            instance=vm_name
        )
        external_ip = vm_info.network_interfaces[0].access_configs[0].nat_i_p
        
        # SSH and execute command
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_key = paramiko.RSAKey.from_private_key_file(ssh_private_key_path)
        
        ssh_client.connect(
            hostname=external_ip,
            username=username,
            pkey=ssh_key,
            timeout=60
        )
        
        stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
        stdout_str = stdout.read().decode()
        stderr_str = stderr.read().decode()
        exit_status = stdout.channel.recv_exit_status()
        
        ssh_client.close()
        
        return (stdout_str, stderr_str, exit_status)


