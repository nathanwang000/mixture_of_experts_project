import os
import signal
import atexit
import subprocess
from itertools import product
import time

# count how many gpu jobs and assign to the most empty one
class GPU_manager():
    def __init__(self, gpus):
        self.gpus = gpus
        self.gpu_jobs = dict((i, 0) for i in gpus)

    def assign_gpu(self):
        gid = sorted(self.gpus, key=lambda gid: self.gpu_jobs[gid])[0]
        self.gpu_jobs[gid] += 1
        return gid

    def rm_job(self, gpu_id):
        self.gpu_jobs[gpu_id] -= 1

def create_command(filename, option_list):
    '''
    example: python filename k1 v1 k2
    INPUTS:
        filename: the python file to run
        option_list: a list of options where each element is either a (k, v) tuple
                     or just a key (to handle "store_const" type of argument);
                     e.g., option_list = [(k1, v1), k2]
        option_dict: (k,v) pairs to handle "python filename --key value" where 
    RETURNS:
        command: ['python', filename, str(k1), str(v1), str(k2)]
    '''
    command = ['python', filename]
    for item in option_list:
        if type(item) in [list, tuple]:
            assert len(item) == 2, "only accept (k,v) pairs for list or tuple argument"
            k, v = item
            command.extend([str(k), str(v)])
        elif type(item) == str:
            command.append(item)
        else:
            raise Exception('unrecognized type, args can only be (k,v) or str')
    return command
    
def run(filename, tasks, gpus, n_concurrent_process):
    '''
    python filename k1 v1 k2
    INPUTS:
        filename: the python file to run
        tasks: list of option_list where option_list is a list of (k,v) pairs or just k
        gpus: list of gpus to use for the tasks
        n_concurrent_process: number of jobs to run in parallel
    RETURNS:
        None; runs the requested jobs in parallel
    '''
    procs = []
    gpu_manager = GPU_manager(gpus)

    commands = []
    for task in tasks:
        commands.append(create_command(filename, task))

    for command in commands:
        print(command)
        my_env = os.environ.copy()
        gid = gpu_manager.assign_gpu()
        my_env['CUDA_VISIBLE_DEVICES'] = str(gid)
        procs.append([subprocess.Popen(command, env=my_env), gid])

        # keep a loop of checking processes
        while True:
            new_procs = []
            while len(procs) > 0:
                p = procs.pop()
                if p[0].poll() == None: # active
                    new_procs.append(p)
                else:
                    gpu_manager.rm_job(p[1])

            procs = new_procs
            if len(procs) >= n_concurrent_process:
                time.sleep(3)
            else:
                break # fetch next
                
    for p in procs:
        p[0].wait()

    def kill_procs():
        for p in procs:
            if p[0].poll() != None:
                pass
            elif p[0].pid is None:
                pass
            else:
                os.kill(p[0].pid, signal.SIGTERM)
        
    atexit.register(kill_procs)

