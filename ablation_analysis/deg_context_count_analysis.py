import os
import pickle

def data_preprocess(deg=15):
    traj_root = "/data1/tpz/nwm-main/data_splits/airvln_16/test"
    traj_file_path = os.path.join(traj_root, f"rollout_turn_{deg}deg.pkl")
    with open(traj_file_path, 'rb') as f:
        traj_data = pickle.load(f)
    
    print(traj_data[29])

if __name__ == "__main__":
    data_preprocess(deg=15)

