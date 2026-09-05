"""Synthetic Shelf Dataset & Ground Truth Generator for Retail Intelligence.

Generates photorealistic shelf benchmark scenes with exact pixel ground truth
for facings, out-of-stock voids, and shelf price tags.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw
import numpy as np

from retail_shelf.config import SAMPLE_DIR, DATA_DIR

def generate_benchmark_scenes():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    catalog_file = DATA_DIR / "store_catalog.json"
    with open(catalog_file, "r", encoding="utf-8") as f:
        catalog = json.load(f)["skus"]
    sku_map = {item["sku_id"]: item for item in catalog}
    
    scenes = []
    
    # Scene 1: Beverage Cooler A3 (Standard busy retail shelf with voids & price tag mismatch)
    scene1 = _render_scene(
        filename="beverages_shelf_01.png",
        title="Beverage Cooler A3 - Peak Hour (OOS on Eye Level)",
        shelf_id="SHELF-BEV-01",
        pog_id="POG-BEV-COOLER-01",
        sku_map=sku_map,
        layout=[
            {"row": 0, "level": "top", "items": [
                {"sku": "SKU-BEV-007", "count": 2, "color": (40, 110, 180)},
                {"sku": "SKU-BEV-003", "count": 3, "color": (230, 130, 40)},
                {"sku": None, "count": 1, "is_void": True, "expected_sku": "SKU-BEV-003"}
            ]},
            {"row": 1, "level": "eye_level", "items": [
                {"sku": "SKU-BEV-001", "count": 2, "color": (20, 50, 130)},
                {"sku": None, "count": 2, "is_void": True, "expected_sku": "SKU-BEV-001"},
                {"sku": "SKU-BEV-002", "count": 4, "color": (30, 30, 30), "accent": (50, 205, 50)}
            ]},
            {"row": 2, "level": "reach", "items": [
                {"sku": "SKU-BEV-004", "count": 4, "color": (210, 30, 30)},
                {"sku": None, "count": 1, "is_void": True, "expected_sku": "SKU-BEV-004"},
                {"sku": "SKU-BEV-005", "count": 3, "color": (160, 160, 160)}
            ]},
            {"row": 3, "level": "bottom", "items": [
                {"sku": "SKU-BEV-006", "count": 3, "color": (10, 80, 170)},
                {"sku": "SKU-BEV-008", "count": 3, "color": (240, 230, 210), "tag_mismatch": 0.69}
            ]}
        ]
    )
    scenes.append(scene1)
    
    # Scene 2: Beverage Cooler A3 (Fully Restocked & Planogram Compliant)
    scene2 = _render_scene(
        filename="beverages_shelf_02_compliant.png",
        title="Beverage Cooler A3 - Fully Restocked (100% Compliant)",
        shelf_id="SHELF-BEV-02",
        pog_id="POG-BEV-COOLER-01",
        sku_map=sku_map,
        layout=[
            {"row": 0, "level": "top", "items": [
                {"sku": "SKU-BEV-007", "count": 2, "color": (40, 110, 180)},
                {"sku": "SKU-BEV-003", "count": 4, "color": (230, 130, 40)}
            ]},
            {"row": 1, "level": "eye_level", "items": [
                {"sku": "SKU-BEV-001", "count": 4, "color": (20, 50, 130)},
                {"sku": "SKU-BEV-002", "count": 4, "color": (30, 30, 30), "accent": (50, 205, 50)}
            ]},
            {"row": 2, "level": "reach", "items": [
                {"sku": "SKU-BEV-004", "count": 5, "color": (210, 30, 30)},
                {"sku": "SKU-BEV-005", "count": 3, "color": (160, 160, 160)}
            ]},
            {"row": 3, "level": "bottom", "items": [
                {"sku": "SKU-BEV-006", "count": 3, "color": (10, 80, 170)},
                {"sku": "SKU-BEV-008", "count": 2, "color": (240, 230, 210)}
            ]}
        ]
    )
    scenes.append(scene2)
    
    # Scene 3: Beverage Cooler A3 (Severe Out of Stock - Dark Shelf Crisis)
    scene3 = _render_scene(
        filename="beverages_shelf_03_depleted.png",
        title="Beverage Cooler A3 - Critical Stockout Crisis",
        shelf_id="SHELF-BEV-03",
        pog_id="POG-BEV-COOLER-01",
        sku_map=sku_map,
        layout=[
            {"row": 0, "level": "top", "items": [
                {"sku": "SKU-BEV-007", "count": 1, "color": (40, 110, 180)},
                {"sku": None, "count": 2, "is_void": True, "expected_sku": "SKU-BEV-007"},
                {"sku": "SKU-BEV-003", "count": 1, "color": (230, 130, 40)},
                {"sku": None, "count": 2, "is_void": True, "expected_sku": "SKU-BEV-003"}
            ]},
            {"row": 1, "level": "eye_level", "items": [
                {"sku": None, "count": 4, "is_void": True, "expected_sku": "SKU-BEV-001"}, # Entire Red Bull row stripped!
                {"sku": "SKU-BEV-002", "count": 1, "color": (30, 30, 30), "accent": (50, 205, 50)},
                {"sku": None, "count": 3, "is_void": True, "expected_sku": "SKU-BEV-002"}
            ]},
            {"row": 2, "level": "reach", "items": [
                {"sku": "SKU-BEV-004", "count": 2, "color": (210, 30, 30)},
                {"sku": None, "count": 3, "is_void": True, "expected_sku": "SKU-BEV-004"},
                {"sku": "SKU-BEV-005", "count": 1, "color": (160, 160, 160)},
                {"sku": None, "count": 2, "is_void": True, "expected_sku": "SKU-BEV-005"}
            ]},
            {"row": 3, "level": "bottom", "items": [
                {"sku": "SKU-BEV-006", "count": 1, "color": (10, 80, 170)},
                {"sku": None, "count": 2, "is_void": True, "expected_sku": "SKU-BEV-006"},
                {"sku": "SKU-BEV-008", "count": 2, "color": (240, 230, 210)}
            ]}
        ]
    )
    scenes.append(scene3)
    
    # Save ground truth file
    gt_file = SAMPLE_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump({"scenes": scenes}, f, indent=2)
    print(f"Generated {len(scenes)} benchmark scenes with ground truth at {gt_file}")
    return scenes

def _render_scene(filename, title, shelf_id, pog_id, sku_map, layout):
    width = 1200
    height = 900
    img = Image.new("RGB", (width, height), (242, 238, 230))
    draw = ImageDraw.Draw(img)
    
    frame_color = (65, 70, 75)
    draw.rectangle([30, 40, 70, height - 30], fill=frame_color)
    draw.rectangle([width - 70, 40, width - 30, height - 30], fill=frame_color)
    
    row_height = 190
    shelf_start_y = 60
    shelf_lip_height = 24
    
    facings_gt = []
    voids_gt = []
    tags_gt = []
    rows_gt = []
    
    for r_idx, row_spec in enumerate(layout):
        row_y = shelf_start_y + r_idx * row_height
        shelf_surface_y = row_y + row_height - shelf_lip_height
        
        draw.rectangle([70, shelf_surface_y, width - 70, shelf_surface_y + shelf_lip_height], fill=(130, 135, 140))
        draw.line([70, shelf_surface_y, width - 70, shelf_surface_y], fill=(200, 205, 210), width=2)
        draw.rectangle([70, shelf_surface_y + 4, width - 70, shelf_surface_y + shelf_lip_height - 2], fill=(45, 48, 52))
        
        row_min_y = row_y + 10
        row_max_y = shelf_surface_y
        rows_gt.append({
            "row_index": r_idx,
            "row_level": row_spec["level"],
            "y_min": row_min_y,
            "y_max": row_max_y + shelf_lip_height
        })
        
        total_items = sum(item["count"] for item in row_spec["items"])
        usable_width = (width - 160)
        item_w = int(usable_width / max(1, total_items))
        item_h = int((row_max_y - row_min_y) * 0.85)
        
        curr_x = 80
        facing_counter = 0
        
        for item in row_spec["items"]:
            count = item["count"]
            is_void = item.get("is_void", False)
            sku_id = item.get("sku")
            sku_info = sku_map.get(sku_id, {}) if sku_id else None
            
            for _ in range(count):
                x1 = curr_x + 6
                x2 = curr_x + item_w - 6
                y2 = shelf_surface_y
                y1 = y2 - item_h
                
                if is_void:
                    void_id = f"VOID_R{r_idx}_{facing_counter}"
                    draw.rectangle([x1, y1, x2, y2], fill=(225, 220, 212), outline=(195, 190, 180), width=1)
                    draw.text((x1 + 10, y1 + item_h // 2 - 10), "EMPTY", fill=(170, 140, 140))
                    
                    voids_gt.append({
                        "id": void_id,
                        "row_index": r_idx,
                        "row_level": row_spec["level"],
                        "bbox": [x1, y1, x2, y2],
                        "expected_sku_id": item.get("expected_sku")
                    })
                else:
                    f_id = f"F_R{r_idx}_{facing_counter}"
                    color = item.get("color", (70, 120, 190))
                    accent = item.get("accent", (255, 255, 255))
                    
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(30, 30, 30), width=2)
                    draw.line([x1 + 4, y1, x1 + 4, y2], fill=(min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), width=3)
                    draw.line([x2 - 4, y1, x2 - 4, y2], fill=(max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40)), width=3)
                    
                    mid_y = y1 + item_h // 2
                    draw.rectangle([x1 + 2, mid_y - 20, x2 - 2, mid_y + 20], fill=accent)
                    
                    display_brand = sku_info["brand"][:8] if sku_info else "BRAND"
                    draw.text((x1 + 8, mid_y - 8), display_brand, fill=(10, 10, 10))
                    
                    tag_w = min(50, item_w - 10)
                    tag_h = 16
                    tag_x1 = x1 + (item_w - tag_w) // 2
                    tag_y1 = shelf_surface_y + 4
                    tag_x2 = tag_x1 + tag_w
                    tag_y2 = tag_y1 + tag_h
                    
                    price_val = item.get("tag_mismatch", sku_info["pos_price"] if sku_info else 1.99)
                    draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
                    draw.text((tag_x1 + 3, tag_y1 + 1), f"${price_val:.2f}", fill=(0, 0, 0))
                    
                    facings_gt.append({
                        "id": f_id,
                        "sku_id": sku_id,
                        "brand": sku_info["brand"] if sku_info else "Unknown",
                        "product_name": sku_info["name"] if sku_info else "Item",
                        "row_index": r_idx,
                        "row_level": row_spec["level"],
                        "bbox": [x1, y1, x2, y2],
                        "price_tag": {
                            "bbox": [tag_x1, tag_y1, tag_x2, tag_y2],
                            "displayed_price": price_val
                        }
                    })
                    
                    tags_gt.append({
                        "sku_id": sku_id,
                        "row_index": r_idx,
                        "displayed_price": price_val,
                        "bbox": [tag_x1, tag_y1, tag_x2, tag_y2]
                    })
                
                curr_x += item_w
                facing_counter += 1
    
    out_path = SAMPLE_DIR / filename
    img.save(out_path, format="PNG")
    print(f"Saved benchmark image {out_path} ({width}x{height})")
    
    return {
        "shelf_id": shelf_id,
        "planogram_id": pog_id,
        "filename": filename,
        "image_width": width,
        "image_height": height,
        "title": title,
        "rows": rows_gt,
        "facings": facings_gt,
        "voids": voids_gt,
        "tags": tags_gt
    }

if __name__ == "__main__":
    generate_benchmark_scenes()
