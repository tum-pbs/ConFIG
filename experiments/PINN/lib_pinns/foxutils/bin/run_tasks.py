#usr/bin/python3

#version:0.0.01
#last modified:20240605
import time
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Manage the scripts commands line by line")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="undo",
        required=False,
        help="The file contains the commands to be executed, default is 'undo'.",
    )  
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="",
        required=False,
        help="Tags for the work, default is empty.",
    )    
    parser.add_argument(
        "-p",
        "--previous", 
        action="store_true", 
        help="Run the previous undo file."
    )
    args = parser.parse_args()
    return args
    
def valid(line):
    if line[0] == "#" or len(line.strip()) ==0:
        return False
    else:
        return True

def num_valid(lines):
    i=0
    for line in lines:
        if valid(line):
            i+=1
    return i

def run_tasks():
    args = parse_args()
    if args.t__tag !="":
        tag="_"+args.t__tag
    else:
        tag=""
    if args.previous:
        previous_folder="./task_records{}/".format(tag)
        if not os.path.exists(previous_folder):
            raise ValueError("No previous work done.")
        dir_list = [folder_name for folder_name in os.listdir(previous_folder) if not os.path.isdir(previous_folder+folder_name)]
        folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(previous_folder, x)))[-1]
        undo_file=os.path.join(previous_folder, folder_name, "done_tasks")
    else:
        undo_file="./{}{}".format(args.f__file,tag)
    timeLabel = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    working_dir="./task_records{}/{}/".format(tag,timeLabel)
    os.makedirs(working_dir,exist_ok=True)
    done_file=os.path.join(working_dir,"done_tasks")
    
    with open(done_file,"w") as fw_done:
        job_idx=1
        while True:
            with open(undo_file,"r") as fr:
                lines=fr.readlines()  
            if len(lines) == 0:
                break      
            for i in range(len(lines)):
                command=lines.pop(0)
                if valid(command):
                    break
                fw_done.write(command)
            with open(undo_file,"w") as fw:     
                for line in lines:
                    fw.write(line.strip()+os.linesep)
            fw.close()  
            log_name=os.path.join(working_dir,"job_{}.log".format(job_idx))
            print('Working on job{}: "{}", {} jobs left'.format(job_idx,command.strip(),num_valid(lines)))
            print("Redirect the output to {}".format(log_name))
            print("")
            os.system(command.strip()+" > {}".format(log_name))
            job_idx+=1
            fw_done.write(command)
    print("All work done. There are no remaining commands in the undo list.")

if __name__ == "__main__":
    run_tasks()   