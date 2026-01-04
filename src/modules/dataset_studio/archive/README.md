# Archive Folder - Alternative Implementations

This folder contains alternative grid implementations and failed experiments that are **no longer used** in production but kept for reference.

## Why These Files Are Archived

These implementations were superseded by the working **FlowLayout** solution in `truscore_dataset_frame_flowlayout.py`.

---

## Failed DearPyGUI Attempts

### 1. fixed_grid_image_cards.py (985 lines)
**Why archived:** DearPyGUI grid attempt with dearpygui-grid library
- **Issues:** Division by zero errors, images overlapping, only 1 column populating
- **Lesson:** dearpygui-grid was too difficult to configure properly
- **Working solution:** PyQt FlowLayout

### 2. fixed_grid_image_cards_backup.py (378 lines)
**Why archived:** Original DearPyGUI demo/backup
- **Status:** Reference implementation

---

## Alternative PyQt Grid Implementations

### 3. truscore_dataset_frame.py (542 lines)
**Why archived:** Older version of dataset frame without FlowLayout
- **Used:** truscore_grid_system.py for grid layout
- **Superseded by:** truscore_dataset_frame_flowlayout.py (with FlowLayout)

### 4. truscore_grid_system.py (361 lines)
**Why archived:** Alternative grid with threading
- **Used by:** truscore_dataset_frame.py
- **Superseded by:** FlowLayout (simpler, more reliable)

### 5. truscore_model_grid.py (419 lines)
**Why archived:** QTableView-based grid with model/view architecture
- **Status:** Alternative implementation, not used in production

### 6. gridlayout.py (134 lines)
**Why archived:** Basic grid layout implementation
- **Status:** Template/reference implementation

---

## Current Production Solution

**File:** `../truscore_dataset_frame_flowlayout.py` (3497 lines)

**Why it works:**
- Uses PyQt's native FlowLayout
- Automatically wraps to new lines
- Responsive to window resizing
- Simple, reliable, no complex grid calculations

**Grid:** `../flowlayout.py` (94 lines)
- Clean FlowLayout implementation
- Left-to-right, top-to-bottom flow
- Automatic wrapping

---

## When to Use These Archives

- **Reference:** If you need to see how alternative grids were implemented
- **Learning:** Understanding what approaches didn't work and why
- **Fallback:** If FlowLayout ever breaks (unlikely), these provide alternatives

## DO NOT USE in Production

These files are archived because:
1. They had bugs/issues
2. They were superseded by better solutions
3. They use deprecated approaches (DearPyGUI for dataset studio)

---

**Archived on:** December 19, 2024  
**Reason:** Systematic cleanup of dataset_creator folder  
**Analysis:** See `../DEPENDENCY_ANALYSIS.json` for full details
