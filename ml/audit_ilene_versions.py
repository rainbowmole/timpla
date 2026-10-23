import os
from roboflow import Roboflow

def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("API Key not found")
        return

    try:
        rf = Roboflow(api_key)
        project = rf.workspace("ilene-cb2s0").project("ingredients-tanjn")
        versions = project.versions()
        
        print(f"Found {len(versions)} versions for project 'ingredients-tanjn':\n")
        
        for v_obj in versions:
            # Attempt to get version number
            try:
                v_num = v_obj.version
            except AttributeError:
                v_num = v_obj.get('version', 'Unknown')
            
            print(f"--- Version {v_num} ---")
            # Try to get the class list for this version.
            # Roboflow's API doesn't always provide classes in the Version object.
            # We might need to download a small sample or look at the project's metadata.
            print(f"Version Object: {v_obj}")
            print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
