import subprocess
import os

def get_gpu_processes(num_gpu):
    # Run nvidia-smi to get the list of processes using the GPUs
    ''' nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
    GPU-bff5b888-68af-f1a4-5534-f97817573a8a, 51356, 1282
    GPU-bff5b888-68af-f1a4-5534-f97817573a8a, 78849, 420
    GPU-bff5b888-68af-f1a4-5534-f97817573a8a, 79415, 610
    GPU-bff5b888-68af-f1a4-5534-f97817573a8a, 27894, 73878
    GPU-36c5b763-b71e-751c-a89c-1afea1823e1e, 51356, 1314
    GPU-55eb09ec-eb66-c0ee-d78c-ddc7bcfd1dea, 51356, 1314
    GPU-55eb09ec-eb66-c0ee-d78c-ddc7bcfd1dea, 30520, 39134
    GPU-b19d5b12-51aa-e585-fe18-1477b4f00153, 51356, 1266
    GPU-b19d5b12-51aa-e585-fe18-1477b4f00153, 31443, 39118
    GPU-6d60cef0-1e72-5a70-bf17-c49861c94a3e, 51356, 1242
    GPU-6d60cef0-1e72-5a70-bf17-c49861c94a3e, 81150, 68156
    GPU-77efb31a-c68c-84ac-e2c4-0d7a3c9df74f, 45782, 39134
    GPU-77efb31a-c68c-84ac-e2c4-0d7a3c9df74f, 51356, 1246
    GPU-2a96e538-da80-621c-f2fb-68332a7766fb, 47144, 39118
    GPU-2a96e538-da80-621c-f2fb-68332a7766fb, 51356, 1246
    GPU-b1fe0c18-be1e-8f8d-d647-9967b0493ea4, 51356, 1222
    GPU-b1fe0c18-be1e-8f8d-d647-9967b0493ea4, 81150, 71506
    '''

    ''' nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader,nounits
    0, GPU-bff5b888-68af-f1a4-5534-f97817573a8a
    1, GPU-36c5b763-b71e-751c-a89c-1afea1823e1e
    2, GPU-55eb09ec-eb66-c0ee-d78c-ddc7bcfd1dea
    3, GPU-b19d5b12-51aa-e585-fe18-1477b4f00153
    4, GPU-6d60cef0-1e72-5a70-bf17-c49861c94a3e
    5, GPU-77efb31a-c68c-84ac-e2c4-0d7a3c9df74f
    6, GPU-2a96e538-da80-621c-f2fb-68332a7766fb
    7, GPU-b1fe0c18-be1e-8f8d-d647-9967b0493ea4
    '''

    ''' fuser -v /dev/nvidia*
                        USER        PID ACCESS COMMAND
    /dev/nvidia0:        root     kernel mount /dev/nvidia0
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia1:        root     kernel mount /dev/nvidia1
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia2:        root     kernel mount /dev/nvidia2
                        jovyan    15444 F...m pt_main_thread
                        jovyan    15824 F...m pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia3:        root     kernel mount /dev/nvidia3
                        jovyan    15444 F...m pt_main_thread
                        jovyan    15824 F...m pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia4:        root     kernel mount /dev/nvidia4
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F...m python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia5:        root     kernel mount /dev/nvidia5
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F...m pt_main_thread
                        jovyan    31778 F...m pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia6:        root     kernel mount /dev/nvidia6
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F.... python
                        jovyan    30934 F...m pt_main_thread
                        jovyan    31778 F...m pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia7:        root     kernel mount /dev/nvidia7
                        jovyan    15444 F.... pt_main_thread
                        jovyan    15824 F.... pt_main_thread
                        jovyan    23520 F...m python
                        jovyan    30934 F.... pt_main_thread
                        jovyan    31778 F.... pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidiactl:      root     kernel mount /dev/nvidiactl
                        jovyan    15444 F...m pt_main_thread
                        jovyan    15824 F...m pt_main_thread
                        jovyan    23520 F...m python
                        jovyan    30934 F...m pt_main_thread
                        jovyan    31778 F...m pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia-uvm:     root     kernel mount /dev/nvidia-uvm
                        jovyan    15444 F...m pt_main_thread
                        jovyan    15824 F...m pt_main_thread
                        jovyan    23520 F...m python
                        jovyan    30934 F...m pt_main_thread
                        jovyan    31778 F...m pt_main_thread
                        jovyan    32407 F...m lighteval
                        jovyan    69714 F...m pt_main_thread
    /dev/nvidia-uvm-tools:
                        root     kernel mount /dev/nvidia-uvm-tools
    '''
        
    all_nvidia_processes = set()
        
    for i in range(num_gpu):
        try:
            fuser_result = subprocess.run(['fuser', '-v', f'/dev/nvidia{i}'], stdout=subprocess.PIPE, text=True, check=True)
            nvidia_devices_output = fuser_result.stdout.strip()
            nvidia_processes = nvidia_devices_output.split(' ')[1:]
            all_nvidia_processes.update(nvidia_processes)
        except subprocess.CalledProcessError as e:
            print(f"运行 fuser -v /dev/nvidia{i} 命令时出错: {e}")
        
    return all_nvidia_processes        


def get_terminal_for_pid(pid):
    try:
        # Get the parent process ID (PPID)
        ppid = int(subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(pid)]).strip())
        # Get the terminal associated with the current and parent process
        terminal_pid = subprocess.check_output(['ps', '-o', 'tty=', '-p', str(pid)]).strip().decode()
        terminal_ppid = subprocess.check_output(['ps', '-o', 'tty=', '-p', str(ppid)]).strip().decode()
        # Get the process command
        cmd_pid = subprocess.check_output(['ps', '-ww', '-o', 'cmd=', '-p', str(pid)]).strip().decode()
        return ppid, cmd_pid, terminal_pid, terminal_ppid
    except subprocess.CalledProcessError:
        return None


def get_mem_usage():
    nvidia_smi_pid = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    for line in nvidia_smi_pid.split('\n'):
        if line:
            gpu_uuid, pid, used_memory = line.split(',')
            gpu_index = get_gpu_index_by_uuid(gpu_uuid)
            print(f"GPU Index: {gpu_index}, PID: {pid}, Used Memory: {used_memory}")
    

def get_gpu_index_by_uuid(gpu_uuid):
    nvidia_smi_index_uuid = subprocess.run(['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    for line in nvidia_smi_index_uuid.split('\n'):
        if gpu_uuid in line:
            return line.split(',')[0]
    return None

        
def main():
    nvidia_processes = get_gpu_processes(8)
    print('=' * 100)
    for pid in nvidia_processes:
        ppid, cmd_pid, terminal_pid, terminal_ppid = get_terminal_for_pid(pid) 
        if str(ppid) in nvidia_processes:
            # print(f"SKIP - PID: {pid} => PPID: {ppid}")
            # print('=' * 100)
            continue
        print(f"PID: {pid} (Terminal: {terminal_pid}) => PPID: {ppid} (Terminal: {terminal_ppid})")
        print(f"\nCommand: \n{cmd_pid}")
        print('-' * 100)
    get_mem_usage()


if __name__ == "__main__":
    main()