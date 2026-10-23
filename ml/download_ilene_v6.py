import os
from roboflow import Roboflow
from pathlib import Path

def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("API Key not found")
        return

    try:
        rf = Roboflow(api_key)
        project = rf.workspace("ilene-cb2s0").project("ingredients-tanjn")
        
        print("Downloading Version 6...")
        dataset = project.version(6).download("yolov8")
        location = Path(dataset.location)
        
        # Save to a dedicated staging directory
        staging_dir = Path("datasets/ilene_v6_staging")
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Provenance record
        provenance = {
            "source_url": "https://universe.roboflow.com/ilene-cb2s0/ingredients-tanjn",
            "creator": "ilene-cb2s0",
            "version": 6,
            "license": "CC BY 4.0",
            "download_date": "2026-10-20",
            "access_method": "Roboflow Python API"
        }
        
        with open(staging_dir / "provenance.json", "w") as f:
            import json
            json.dump(provenance, f, indent=2)
            
        # Check the class list in the downloaded data.yaml
        data_yaml_path = location / "data.yaml"
        if data_yaml_path.exists():
            import yaml
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                
                with open(staging_dir / "full_class_list.txt", "w") as f_out:
                    if isinstance(names, dict):
                        for i in sorted(names.keys()):
                            f_out.write(f"{i}: {names[i]}\n")
                    else:
                        for i, name in enumerate(names):
                            f_out.write(f"{i}: {name}\n")
                print(f"Successfully extracted {len(names)} classes.")
        else:
            print("Could not find data.yaml in downloaded dataset.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
