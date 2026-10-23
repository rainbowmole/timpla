import os
import shutil
import yaml
from ultralytics import YOLO
from pathlib import Path

def load_yaml_names(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return names

def run_remapped_eval(model_path, current_data_yaml, v1_list_path):
    current_names = load_yaml_names(current_data_yaml)
    model = YOLO(model_path)
    model_names = model.names # {index: name}
    
    curr_to_model = {}
    for m_idx, m_name in model_names.items():
        try:
            c_idx = current_names.index(m_name)
            curr_to_model[c_idx] = m_idx
        except ValueError:
            continue

    # Use absolute paths for temp directory to avoid Ultralytics path issues
    base_path = Path(os.getcwd()).absolute()
    temp_dir = base_path / f"temp_eval_{os.path.basename(model_path).split('/')[-1]}"
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / "images").mkdir(exist_ok=True)
    (temp_dir / "labels").mkdir(exist_ok=True)
    
    with open(v1_list_path, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]

    for img_p in img_paths:
        abs_img_p = Path(img_p).absolute()
        if not abs_img_p.exists(): continue
        
        # Copy image
        shutil.copy(abs_img_p, temp_dir / "images" / abs_img_p.name)
        
        # Find label: search timpla_combined/valid/labels for a file matching the image stem
        img_stem = abs_img_p.stem
        # Roboflow labels can be complex. Try exact match first, then search.
        lbl_src = Path("timpla_combined/valid/labels") / (img_stem + ".txt")
        if not lbl_src.exists():
            for l_file in (Path("timpla_combined/valid/labels")).glob("*.txt"):
                if img_stem in l_file.name:
                    lbl_src = l_file
                    break
        
        if lbl_src.exists():
            with open(lbl_src, 'r') as f_in, open(temp_dir / "labels" / (lbl_src.name), 'w') as f_out:
                for line in f_in:
                    parts = line.split()
                    if not parts: continue
                    c_idx = int(parts[0])
                    if c_idx in curr_to_model:
                        m_idx = curr_to_model[c_idx]
                        f_out.write(f"{m_idx} {' '.join(parts[1:])}\n")

    # Create temp data.yaml with ABSOLUTE paths
    temp_data_yaml = temp_dir / "data.yaml"
    with open(temp_data_yaml, 'w') as f:
        f.write(f"train: {temp_dir}/images\nval: {temp_dir}/images\ntest: {temp_dir}/images\n")
        f.write(f"nc: {len(model_names)}\nnames:\n")
        for i in range(len(model_names)):
            f.write(f"- {model_names[i]}\n")

    # Run validation
    results = model.val(data=str(temp_data_yaml), imgsz=640, device="mps", plots=False, verbose=True)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ml/remap_and_eval.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    data_yaml = "timpla_combined/data.yaml"
    v1_list = "timpla_combined/valid_v1.txt"
    
    res = run_remapped_eval(model_path, data_yaml, v1_list)
    print(f"\nFINAL RESULTS for {model_path}:")
    print(f"mAP50: {res.box.map50:.4f}")
    print(f"mAP50-95: {res.box.map:.4f}")
