import os
import json
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
        
        # List versions
        versions = project.versions()
        if not versions:
            print("No versions found.")
            return
        
        # Extract the version number from the last Version object
        latest_version_obj = versions[-1]
        # Based on the error, the object is a dict-like object or has a .version attribute
        # We will try to get the version number.
        try:
            latest_version = latest_version_obj.version
        except AttributeError:
            # If it's a dict
            latest_version = latest_version_obj.get('version') if hasattr(latest_version_obj, 'get') else latest_version_obj
            
        print(f"Downloading version {latest_version}...")
        dataset = project.version(latest_version).download("yolov8")
        location = Path(dataset.location)
        
        # Provenance metadata
        provenance = {
            "source_url": "https://universe.roboflow.com/ilene-cb2s0/ingredients-tanjn",
            "creator": "ilene-cb2s0",
            "version": latest_version,
            "license": "CC BY 4.0",
            "download_date": "2026-10-20",
            "access_method": "Roboflow Python API"
        }
        
        # Save provenance and location
        audit_dir = Path("datasets/ilene_raw_audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        with open(audit_dir / "provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
        
        # Extract the class list from the downloaded data.yaml
        data_yaml_path = location / "data.yaml"
        if data_yaml_path.exists():
            import yaml
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                names = data.get('names', [])
                
                # Write the class list to a file for the audit
                with open(audit_dir / "full_class_list.txt", "w") as f_out:
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
