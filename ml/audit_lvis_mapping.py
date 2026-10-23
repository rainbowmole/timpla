import csv
import yaml
from ml.merge_datasets import CANONICAL_CLASSES

def audit_lvis():
    canonical = set(CANONICAL_CLASSES)
    lvis_counts_path = "timpla-kyle_ver/datasets/reports/produce-class-counts.csv"
    
    print(f"{'LVIS Class':<30} | {'Action':<20} | {'Canonical Class':<20}")
    print("-" * 75)
    
    with open(lvis_counts_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lvis_class = row['class'].lower().strip()
            # Handle multi-name classes like "bell pepper/capsicum"
            parts = lvis_class.split('/')
            
            mapped = False
            for p in parts:
                p = p.strip()
                if p in canonical:
                    print(f"{lvis_class:<30} | map_to_existing     | {p:<20}")
                    mapped = True
                    break
            
            if not mapped:
                # Special handling for sili/chili
                if 'chili' in lvis_class or 'cayenne' in lvis_class:
                    # Based on la-LVIS names: "cayenne/cayenne spice..." and "chili/chili vegetable..."
                    # We map to generic 'chili' as we can't distinguish haba/labuyo from LVIS text alone
                    print(f"{lvis_class:<30} | map_to_existing     | chili")
                else:
                    print(f"{lvis_class:<30} | exclude             | N/A")

if __name__ == "__main__":
    audit_lvis()
