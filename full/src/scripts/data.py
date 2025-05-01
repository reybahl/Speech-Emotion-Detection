import subprocess
import os
import numpy as np

ORIGINAL_PATH = "crema-d-mirror"
SUBDIRECTORY = "AudioMP3"
NEW_PATH = "data"

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

def load_data():
    repo_url = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"
    branch = "main"

    current_dir = os.getcwd()
    
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    repo_path = os.path.join(current_dir, repo_name)
    
    # Check if repository already exists
    if os.path.exists(repo_path):
        print(f"Repository already exists at {repo_path}")
    else:
        # Clone the repository
        try:
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            raise
    
    os.chdir(repo_path)

    try:
        if not os.path.exists(os.path.join(repo_path, ".git", "info", "sparse-checkout")):
            subprocess.run(["git", "sparse-checkout", "init", "--cone"], check=True)
            subprocess.run(["git", "sparse-checkout", "set", SUBDIRECTORY], check=True)
            subprocess.run(["git", "checkout", branch], check=True)
        else:
            print("Sparse checkout already initialized")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up sparse checkout: {e}")
        raise
    
    os.chdir(current_dir)
    
    return repo_path


"""
Example file name:
1001_DFA_ANG_XX.mp3

Notes from the GitHub repo:

The sentences were presented using different emotion (in parentheses is the three letter code used in the third part of the filename):

Anger (ANG)
Disgust (DIS)
Fear (FEA)
Happy/Joy (HAP)
Neutral (NEU)
Sad (SAD)

"""
def organize_data():
    repo_path = load_data()
    
    # Construct the full path to AudioMP3
    audio_path = os.path.join(repo_path, SUBDIRECTORY)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"AudioMP3 directory not found at {audio_path}")
    
    files = os.listdir(audio_path)
    print(f"Found {len(files)} files in {audio_path}")

    # randomly split into training (80%), val (10%), test (10%)
    train_files, val_files, test_files = np.split(files, [int(.8 * len(files)), int(.9 * len(files))])
        
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    os.makedirs(NEW_PATH, exist_ok=True)

    # move the files to their corresponding subdirectories
    for split in splits.keys():
        for emotion in EMOTIONS:
            os.makedirs(os.path.join(NEW_PATH, split, emotion), exist_ok=True)

        files = splits[split]
        split_path = os.path.join(NEW_PATH, split)

        for file in files:
            # get the emotion from the file name
            emotion = file.split('_')[2]
            if emotion in EMOTIONS:
                src_path = os.path.join(audio_path, file)
                dst_path = os.path.join(split_path, emotion, file)
                os.rename(src_path, dst_path)
            else:
                raise ValueError(f"Invalid emotion: {emotion}")

if __name__ == "__main__":
    try: 
        organize_data()
    except Exception as e:
        print(f"Error: {e}")
