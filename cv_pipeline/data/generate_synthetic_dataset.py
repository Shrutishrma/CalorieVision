import json, math, random
from pathlib import Path

CLASSES = ["squat", "pushup", "plank", "jumping_jack", "lunge", "situp", "burpee", "mountain_climber", "rest"]
SEQ_LEN = 15
NUM_SEQS = 400

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

def make_skeleton(
    head_pos=(0.5, 0.1),
    l_sh=(0.6, 0.2), r_sh=(0.4, 0.2),
    l_el=(0.65, 0.35), r_el=(0.35, 0.35),
    l_wr=(0.65, 0.45), r_wr=(0.35, 0.45),
    l_hip=(0.55, 0.5), r_hip=(0.45, 0.5),
    l_kn=(0.55, 0.75), r_kn=(0.45, 0.75),
    l_ank=(0.55, 0.95), r_ank=(0.45, 0.95)
):
    lms = []
    for i, name in enumerate(LANDMARK_NAMES):
        if i == 0:
            x, y = head_pos
        elif i in (1, 2, 3, 7, 9):
            x, y = head_pos[0] + 0.02, head_pos[1]
        elif i in (4, 5, 6, 8, 10):
            x, y = head_pos[0] - 0.02, head_pos[1]
        elif i == 11:
            x, y = l_sh
        elif i == 12:
            x, y = r_sh
        elif i == 13:
            x, y = l_el
        elif i == 14:
            x, y = r_el
        elif i == 15:
            x, y = l_wr
        elif i == 16:
            x, y = r_wr
        elif i in (17, 19, 21):
            x, y = l_wr[0] + 0.02, l_wr[1] + 0.02
        elif i in (18, 20, 22):
            x, y = r_wr[0] - 0.02, r_wr[1] + 0.02
        elif i == 23:
            x, y = l_hip
        elif i == 24:
            x, y = r_hip
        elif i == 25:
            x, y = l_kn
        elif i == 26:
            x, y = r_kn
        elif i == 27:
            x, y = l_ank
        elif i == 28:
            x, y = r_ank
        elif i in (29, 31):
            x, y = l_ank[0], l_ank[1] + 0.03
        elif i in (30, 32):
            x, y = r_ank[0], r_ank[1] + 0.03
        else:
            x, y = 0.5, 0.5

        lms.append({
            "index": i,
            "name": name,
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "visibility": 0.99
        })

    return {
        "frame_index": 0,
        "timestamp": 0.0,
        "pose_detected": True,
        "landmarks": lms
    }

def noise(val, amt=0.015):
    return val + random.uniform(-amt, amt)

def gen_squat(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(math.pi * i / n) # 0 -> 1 -> 0
        hip_y = noise(0.5 + 0.25 * p)
        knee_x_offset = 0.08 * p
        knee_y = noise(0.72 + 0.08 * p)
        sh_y = noise(0.2 + 0.20 * p)
        head_y = noise(0.1 + 0.20 * p)
        frames.append(make_skeleton(
            head_pos=(0.5, head_y),
            l_sh=(0.6, sh_y), r_sh=(0.4, sh_y),
            l_el=(0.65, sh_y + 0.15), r_el=(0.35, sh_y + 0.15),
            l_wr=(0.65, sh_y + 0.25), r_wr=(0.35, sh_y + 0.25),
            l_hip=(0.55, hip_y), r_hip=(0.45, hip_y),
            l_kn=(0.55 + knee_x_offset, knee_y), r_kn=(0.45 - knee_x_offset, knee_y),
            l_ank=(0.55, 0.95), r_ank=(0.45, 0.95)
        ))
    return frames

def gen_pushup(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(math.pi * i / n) # down and up
        sh_y = noise(0.75 + 0.15 * p)
        hip_y = noise(0.70 + 0.12 * p)
        el_y = noise(0.70 + 0.20 * p)
        frames.append(make_skeleton(
            head_pos=(0.78, sh_y - 0.05),
            l_sh=(0.70, sh_y), r_sh=(0.70, sh_y),
            l_el=(0.70, el_y), r_el=(0.70, el_y),
            l_wr=(0.70, 0.92), r_wr=(0.70, 0.92),
            l_hip=(0.45, hip_y), r_hip=(0.45, hip_y),
            l_kn=(0.30, 0.85), r_kn=(0.30, 0.85),
            l_ank=(0.15, 0.90), r_ank=(0.15, 0.90)
        ))
    return frames

def gen_plank(n=SEQ_LEN):
    frames = []
    for i in range(n):
        sh_y = noise(0.75, 0.005)
        hip_y = noise(0.72, 0.005)
        frames.append(make_skeleton(
            head_pos=(0.78, sh_y - 0.05),
            l_sh=(0.70, sh_y), r_sh=(0.70, sh_y),
            l_el=(0.70, 0.90), r_el=(0.70, 0.90),
            l_wr=(0.78, 0.90), r_wr=(0.78, 0.90),
            l_hip=(0.45, hip_y), r_hip=(0.45, hip_y),
            l_kn=(0.30, 0.82), r_kn=(0.30, 0.82),
            l_ank=(0.15, 0.88), r_ank=(0.15, 0.88)
        ))
    return frames

def gen_jumping_jack(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(2 * math.pi * i / n)
        # arms up, legs wide
        arm_spread = 0.35 * abs(p)
        arm_up = 0.35 * abs(p)
        leg_spread = 0.15 * abs(p)
        frames.append(make_skeleton(
            head_pos=(0.5, 0.1),
            l_sh=(0.6, 0.2), r_sh=(0.4, 0.2),
            l_el=(0.65 + arm_spread * 0.5, 0.35 - arm_up * 0.5),
            r_el=(0.35 - arm_spread * 0.5, 0.35 - arm_up * 0.5),
            l_wr=(0.65 + arm_spread, 0.45 - arm_up),
            r_wr=(0.35 - arm_spread, 0.45 - arm_up),
            l_hip=(0.55, 0.5), r_hip=(0.45, 0.5),
            l_kn=(0.55 + leg_spread * 0.5, 0.75),
            r_kn=(0.45 - leg_spread * 0.5, 0.75),
            l_ank=(0.55 + leg_spread, 0.95),
            r_ank=(0.45 - leg_spread, 0.95)
        ))
    return frames

def gen_lunge(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(math.pi * i / n)
        hip_y = noise(0.55 + 0.15 * p)
        front_knee_y = noise(0.70 + 0.10 * p)
        back_knee_y = noise(0.75 + 0.15 * p)
        frames.append(make_skeleton(
            head_pos=(0.5, 0.15 + 0.1 * p),
            l_sh=(0.55, 0.25 + 0.1 * p), r_sh=(0.45, 0.25 + 0.1 * p),
            l_el=(0.58, 0.40 + 0.1 * p), r_el=(0.42, 0.40 + 0.1 * p),
            l_wr=(0.58, 0.50 + 0.1 * p), r_wr=(0.42, 0.50 + 0.1 * p),
            l_hip=(0.52, hip_y), r_hip=(0.48, hip_y),
            l_kn=(0.65, front_knee_y), r_kn=(0.35, back_knee_y),
            l_ank=(0.65, 0.95), r_ank=(0.25, 0.90)
        ))
    return frames

def gen_situp(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(math.pi * i / n) # 0=lying down, 1=sitting up
        sh_x = noise(0.25 + 0.25 * p)
        sh_y = noise(0.85 - 0.40 * p)
        head_x = noise(0.18 + 0.25 * p)
        head_y = noise(0.85 - 0.45 * p)
        frames.append(make_skeleton(
            head_pos=(head_x, head_y),
            l_sh=(sh_x, sh_y), r_sh=(sh_x, sh_y),
            l_el=(sh_x + 0.05, sh_y + 0.1), r_el=(sh_x + 0.05, sh_y + 0.1),
            l_wr=(sh_x + 0.08, sh_y), r_wr=(sh_x + 0.08, sh_y),
            l_hip=(0.55, 0.88), r_hip=(0.55, 0.88),
            l_kn=(0.68, 0.70), r_kn=(0.68, 0.70),
            l_ank=(0.78, 0.92), r_ank=(0.78, 0.92)
        ))
    return frames

def gen_burpee(n=SEQ_LEN):
    frames = []
    phases = [0]*4 + [1]*6 + [2]*5
    for i in range(n):
        ph = phases[i]
        if ph == 0:
            frames.extend(gen_squat(1))
        elif ph == 1:
            frames.extend(gen_pushup(1))
        else:
            frames.extend(gen_jumping_jack(1))
    return frames[:n]

def gen_mountain_climber(n=SEQ_LEN):
    frames = []
    for i in range(n):
        p = math.sin(2 * math.pi * i / n)
        l_kn_x = 0.30 + 0.18 * max(0, p)
        r_kn_x = 0.30 + 0.18 * max(0, -p)
        frames.append(make_skeleton(
            head_pos=(0.78, 0.70),
            l_sh=(0.70, 0.75), r_sh=(0.70, 0.75),
            l_el=(0.70, 0.85), r_el=(0.70, 0.85),
            l_wr=(0.70, 0.92), r_wr=(0.70, 0.92),
            l_hip=(0.45, 0.72), r_hip=(0.45, 0.72),
            l_kn=(l_kn_x, 0.82), r_kn=(r_kn_x, 0.82),
            l_ank=(0.15, 0.90), r_ank=(0.15, 0.90)
        ))
    return frames

def gen_rest(n=SEQ_LEN):
    frames = []
    for i in range(n):
        frames.append(make_skeleton(
            head_pos=(noise(0.5, 0.005), noise(0.1, 0.005)),
            l_sh=(noise(0.6, 0.005), noise(0.2, 0.005)), r_sh=(noise(0.4, 0.005), noise(0.2, 0.005)),
            l_el=(noise(0.65, 0.005), noise(0.35, 0.005)), r_el=(noise(0.35, 0.005), noise(0.35, 0.005)),
            l_wr=(noise(0.65, 0.005), noise(0.45, 0.005)), r_wr=(noise(0.35, 0.005), noise(0.45, 0.005)),
            l_hip=(noise(0.55, 0.005), noise(0.5, 0.005)), r_hip=(noise(0.45, 0.005), noise(0.5, 0.005)),
            l_kn=(noise(0.55, 0.005), noise(0.75, 0.005)), r_kn=(noise(0.45, 0.005), noise(0.75, 0.005)),
            l_ank=(noise(0.55, 0.005), noise(0.95, 0.005)), r_ank=(noise(0.45, 0.005), noise(0.95, 0.005))
        ))
    return frames

generators = {
    "squat": gen_squat, "pushup": gen_pushup, "plank": gen_plank,
    "jumping_jack": gen_jumping_jack, "lunge": gen_lunge, "situp": gen_situp,
    "burpee": gen_burpee, "mountain_climber": gen_mountain_climber, "rest": gen_rest
}

all_frames, all_labels = [], []
for label, func in generators.items():
    print(f"  Generating {NUM_SEQS} sequences for {label}...")
    for _ in range(NUM_SEQS):
        frames = func(SEQ_LEN)
        for idx, f in enumerate(frames):
            f["frame_index"] = len(all_frames)
            f["timestamp"] = len(all_frames) * 0.5
            all_frames.append(f)
            all_labels.append(label)

Path("dataset").mkdir(exist_ok=True)
with open("dataset/train_data.json", "w", encoding="utf-8") as f:
    json.dump({"frames": all_frames}, f)
with open("dataset/train_labels.json", "w", encoding="utf-8") as f:
    json.dump(all_labels, f)

print(f"Dataset generated: {len(all_frames)} frames across {len(generators)} classes.")
