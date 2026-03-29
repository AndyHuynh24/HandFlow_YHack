"""Generate foldable A4 PDF with ArUco-marked macropad sets for origami-style stand."""

import os
import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Fallback values (A4 in points: 595 x 842)
    A4 = (595.276, 841.89)
    mm = 2.834645669291339
    cm = 28.346456692913385
    colors = None
    canvas = None


@dataclass
class MacroPadSetInfo:
    """Info for a macropad set to print."""
    name: str
    set_marker_id: int
    button_labels: List[str]  # 12 labels for buttons 0-11


class OrigamiMacroPadPDFGenerator:
    """
    Generate origami-foldable A4 PDF with 2 macropad sets.

    Features:
    - 8 corner markers per set (TL=set ID, others constant 4-10)
    - BL2/BR2 provide fallback detection when BL/BR are occluded
    - 4x3 button grid with 70% circle indicators (12 buttons per set)
    - Fold guide lines
    - Square buttons for better usability
    """
    
    # A4 dimensions
    PAGE_WIDTH, PAGE_HEIGHT = A4  # 210mm x 297mm in points
    
    # Margins in mm
    TOP_MARGIN_MM = 21
    SIDE_MARGIN_MM = 25
    INFO_MARGIN_MM = 24  # Bottom margin within each section for info
    
    # Constant marker IDs (positions TR, ML, MR, BL, BR, BL2, BR2)
    CONSTANT_MARKER_IDS = [4, 5, 6, 7, 8, 9, 10]
    
    # Grid layout
    GRID_COLS = 4
    GRID_ROWS = 3
    CIRCLE_RATIO = 0.7  # Circle diameter as fraction of cell size

    # Fold line offset: move fold guides into margin (away from content/markers)
    FOLD_LINE_OFFSET_MM = 1.5
    
    def __init__(self, aruco_dict=cv2.aruco.DICT_4X4_50):
        """Initialize generator."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation.\n"
                "Install with: pip install reportlab"
            )
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._temp_files = []
    
    def generate(
        self,
        sets: List[MacroPadSetInfo],
        output_path: str
    ) -> str:
        """
        Generate PDF with up to 2 macropad sets (one per page section).

        Args:
            sets: List of MacroPadSetInfo (max 2)
            output_path: Path to save PDF

        Returns:
            Path to generated PDF
        """
        if len(sets) > 2:
            sets = sets[:2]
        while len(sets) < 2:
            # Pad with empty sets if needed
            sets.append(MacroPadSetInfo(
                name=f"Set {len(sets) + 1}",
                set_marker_id=12 + len(sets),
                button_labels=[f"Button {i+1}" for i in range(12)]
            ))

        c = canvas.Canvas(output_path, pagesize=A4)

        # Calculate section dimensions
        page_width_mm = 210
        page_height_mm = 297

        top_margin = self.TOP_MARGIN_MM * mm
        side_margin = self.SIDE_MARGIN_MM * mm
        info_margin = self.INFO_MARGIN_MM * mm

        # Available height for 2 sections (larger sections for square buttons)
        available_height = self.PAGE_HEIGHT - top_margin
        section_height = available_height / 2

        # Content area within margins
        content_width = self.PAGE_WIDTH - 2 * side_margin

        # Draw each section
        for idx, set_info in enumerate(sets):
            section_top = self.PAGE_HEIGHT - top_margin - idx * section_height

            self._draw_section(
                c, set_info,
                x=side_margin,
                y=section_top - section_height,
                width=content_width,
                height=section_height,
                info_margin=info_margin,
                flip_info=False  # No flipping for any section
            )

            # Draw fold guide line between sections (except after last)
            # Offset into margin region (away from markers/content)
            if idx < 1:
                fold_offset = self.FOLD_LINE_OFFSET_MM * mm
                fold_y = section_top - section_height
                self._draw_fold_line(c, 0, fold_y, self.PAGE_WIDTH)

        # Draw fold guides on margins (except info margin)
        self._draw_margin_fold_guides(c, top_margin, side_margin, section_height)

        c.save()
        self._cleanup_temp_files()

        print(f"[PDF] Origami macropad generated: {output_path}")
        return output_path
    
    def _draw_section(
        self,
        c,
        set_info: MacroPadSetInfo,
        x: float,
        y: float,
        width: float,
        height: float,
        info_margin: float,
        flip_info: bool
    ):
        """Draw a single macropad section."""
        # Marker region height (section minus info margin)
        fold_offset = self.FOLD_LINE_OFFSET_MM * mm
        marker_region_height = height - fold_offset * 4
        
        # Same layout for all sections: markers at top, info at bottom
        info_y = y
        marker_y = y + info_margin
        
        # Draw markers and grid
        self._draw_markers_and_grid(
            c, set_info,
            x, y + fold_offset * 3,
            width, marker_region_height
        )
        
        # Draw info area (text flipped for bottom section)
        self._draw_info_area(
            c, set_info,
            x, info_y,
            width, info_margin,
            flip_info
        )
    
    def _draw_markers_and_grid(
        self,
        c,
        set_info: MacroPadSetInfo,
        x: float,
        y: float,
        width: float,
        height: float
    ):
        """
        Draw 8 markers and 4x3 button grid with square buttons.

        Marker layout:
            [TL: Set ID]              [TR: ID 4]
                   |                       |
            [ML: ID 5]                [MR: ID 6]
                   |                       |
            [BL2: ID 9]              [BR2: ID 10]
                   | 0 gap                 | 0 gap
                   [BL: ID 7]    [BR: ID 8]  <- inner bottom markers

        BL2 and BR2 provide fallback bottom corners when BL/BR are occluded.
        """
        # Calculate marker size for square buttons in 4x3 grid
        # For square buttons: grid_width/4 = grid_height/3
        # With grid_width = width - 2*M and grid_height = 4*M (height = 5*M)
        # Solving: (width - 2*M)/4 = 4*M/3 => M = 3*width/22
        marker_size = (3 * width) / 22

        # Ensure minimum marker size for detection
        min_marker_size = 15 * mm
        marker_size = max(marker_size, min_marker_size)

        # Gap for visual border drawing only (borders stop before reaching markers)
        # This does NOT affect the detection area - just the printed borders
        border_gap = 2 * mm  # 2mm visual gap near markers

        # Marker positions (4 on left edge, 4 on right edge)
        left_x = x
        right_x = x + width - marker_size

        # Distribute markers evenly across height (height = 5 * marker_size)
        # TL/TR at top, ML/MR in middle, BL2/BR2 below middle, BL/BR at bottom
        top_y = y + height - marker_size
        mid_y = top_y - marker_size * 1.5
        bottom2_y = mid_y - marker_size * 1.5  # BL2/BR2 directly under ML/MR
        bottom_y = bottom2_y - marker_size

        # Position map: (position_name, marker_id, x, y)
        positions = [
            ('TL', set_info.set_marker_id, left_x, top_y),
            ('TR', 4, right_x, top_y),
            ('ML', 5, left_x, mid_y),
            ('MR', 6, right_x, mid_y),
            ('BL2', 9, left_x, bottom2_y),           # directly under ML
            ('BR2', 10, right_x, bottom2_y),         # directly under MR
            ('BL', 7, left_x + marker_size, bottom_y),  # inner bottom left
            ('BR', 8, right_x - marker_size, bottom_y), # inner bottom right
        ]

        # Draw markers
        for pos_name, marker_id, mx, my in positions:
            self._draw_marker(c, marker_id, mx, my, marker_size)

        # Grid region: matches detection region (from TL's inner corner to BL2/BR2's outer corner)
        # Detection region: TL's top-right to BR2's bottom-left
        grid_x = left_x + marker_size  # Starts at TL's right edge
        grid_y = bottom2_y  # Starts at BL2's bottom edge (detection region bottom)
        grid_width = right_x - grid_x  # Full width to TR's left edge
        grid_height = (top_y + marker_size) - bottom2_y  # From BL2 bottom to TL top

        # Draw 4x3 grid with buttons
        # Pass border_gap so borders don't touch markers visually
        self._draw_button_grid(c, set_info.button_labels, grid_x, grid_y, grid_width, grid_height, border_gap, marker_size)
    
    def _draw_marker(self, c, marker_id: int, x: float, y: float, size: float):
        """Draw a single ArUco marker."""
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, 200)
        temp_path = f"/tmp/aruco_macropad_{marker_id}_{id(self)}.png"
        cv2.imwrite(temp_path, marker_img)
        self._temp_files.append(temp_path)
        
        # Draw to PDF
        c.drawImage(
            temp_path,
            x, y,
            width=size,
            height=size,
            preserveAspectRatio=True
        )
        
        # Draw ID label
        # c.setFont("Helvetica-Bold", 6)
        # c.setFillColor(colors.black)
        # c.drawCentredString(x + size / 2, y - 8, f"ID {marker_id}")
    
    def _draw_button_grid(
        self,
        c,
        labels: List[str],
        x: float,
        y: float,
        width: float,
        height: float,
        border_gap: float = 0,
        marker_size: float = 0
    ):
        """
        Draw 4x3 button grid with circles and labels.

        Borders have gaps near the left/right edges so they don't touch markers.
        """
        cell_width = width / self.GRID_COLS
        cell_height = height / self.GRID_ROWS
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                idx = row * self.GRID_COLS + col
                label = labels[idx] if idx < len(labels) else f"Btn {idx + 1}"
                
                cell_x = x + col * cell_width
                cell_y = y + height - (row + 1) * cell_height  # Top-down
                
                # Determine if cell is at left or right edge
                is_left_edge = (col == 0)
                is_right_edge = (col == self.GRID_COLS - 1)

                is_left_bottom_edge = is_left_edge and (row == self.GRID_ROWS - 1)
                is_right_bottom_edge = is_right_edge and (row == self.GRID_ROWS - 1)
                
                # Draw cell borders with gaps at edges
                # Top border
                top_y_line = cell_y + cell_height
                left_start = cell_x + (border_gap if is_left_edge else 0)
                right_end = cell_x + cell_width - (border_gap if is_right_edge else 0)
                c.line(left_start, top_y_line, right_end, top_y_line)
                
                # Bottom border
                bottom_y_line = cell_y
                if is_left_bottom_edge :
                    c.line(left_start + marker_size, bottom_y_line, right_end, bottom_y_line)
                elif is_right_bottom_edge :
                    c.line(left_start, bottom_y_line, right_end - marker_size, bottom_y_line)
                else: 
                    c.line(left_start, bottom_y_line, right_end, bottom_y_line)
                
                # Left border (only internal ones, not at left edge)
                if not is_left_edge:
                    c.line(cell_x, cell_y, cell_x, cell_y + cell_height)
                
                # Right border (only internal ones, not at right edge)
                if not is_right_edge:
                    c.line(cell_x + cell_width, cell_y, cell_x + cell_width, cell_y + cell_height)
                
                # Circle indicator (70% of cell size)
                circle_diameter = min(cell_width, cell_height) * self.CIRCLE_RATIO
                circle_x = cell_x + cell_width / 2
                circle_y = cell_y + cell_height / 2
                
                c.setStrokeColor(colors.Color(0.3, 0.3, 0.3))
                c.setLineWidth(1)
                c.circle(circle_x, circle_y, circle_diameter / 2, stroke=1, fill=0)
                
                # # Button label inside circle
                # c.setFont("Helvetica-Bold", 8)
                # c.setFillColor(colors.black)
                
                # # Truncate long labels
                # if len(label) > 10:
                #     label = label[:9] + ".."
                
                # c.drawCentredString(circle_x, circle_y - 3, label)
                
                # Button number (small, corner)
                c.setFont("Helvetica", 5)
                c.setFillColor(colors.Color(0.5, 0.5, 0.5))
                c.drawString(cell_x + 2, cell_y + cell_height - 8, str(idx + 1))
                
                # Reset stroke color for next cell
                c.setStrokeColor(colors.black)
    
    def _draw_info_area(
        self,
        c,
        set_info: MacroPadSetInfo,
        x: float,
        y: float,
        width: float,
        height: float,
        flip: bool
    ):
        """Draw info area with set name and instructions."""
        if flip:
            # Save state and rotate
            c.saveState()
            c.translate(x + width / 2, y + height / 2)
            c.rotate(180)
            c.translate(-width / 2, -height / 2)
            
            # Draw content at origin
            self._draw_info_content(c, set_info, 0, 0, width, height)
            
            c.restoreState()
        else:
            self._draw_info_content(c, set_info, x, y, width, height)
    
    def _draw_info_content(
        self,
        c,
        set_info: MacroPadSetInfo,
        x: float,
        y: float,
        width: float,
        height: float
    ):
        """Draw info content (set name, marker ID, instructions)."""
        center_x = x + width / 2
        text_y = y + height / 2
        
        # Set name
        # c.setFont("Helvetica-Bold", 12)
        # c.setFillColor(colors.black)
        # c.drawCentredString(center_x, text_y + 5, set_info.name)
        
        # Marker ID info
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.Color(0.4, 0.4, 0.4))
        c.drawCentredString(center_x, text_y - 8, f"Set ID: {set_info.set_marker_id}")
    
    def _draw_fold_line(self, c, x: float, y: float, width: float):
        """Draw horizontal fold guide line."""
        c.setStrokeColor(colors.Color(0.7, 0.7, 0.7))
        c.setLineWidth(0.5)
        c.setDash(3, 3)
        c.line(x, y, x + width, y)
        c.setDash()
    
    def _draw_margin_fold_guides(
        self,
        c,
        top_margin: float,
        side_margin: float,
        section_height: float
    ):
        """Draw fold guide lines on margins, offset 1.5mm into margin (away from content)."""
        fold_offset = self.FOLD_LINE_OFFSET_MM * mm

        c.setStrokeColor(colors.Color(0.7, 0.7, 0.7))
        c.setLineWidth(0.5)
        c.setDash(3, 3)

        # Top edge - offset UP into margin (away from content)
        fold_y = self.PAGE_HEIGHT - top_margin 
        c.line(0, fold_y, self.PAGE_WIDTH, fold_y)

        # Left margin (vertical) - offset LEFT into margin (away from content)
        left_x = side_margin - fold_offset * 2
        c.line(left_x, 0, left_x, self.PAGE_HEIGHT)

        # Right margin (vertical) - offset RIGHT into margin (away from content)
        right_x = self.PAGE_WIDTH - side_margin + fold_offset * 2
        c.line(right_x, 0, right_x, self.PAGE_HEIGHT)

        c.setDash()
    
    def _cleanup_temp_files(self):
        """Remove temporary marker images."""
        for path in self._temp_files:
            try:
                os.remove(path)
            except:
                pass
        self._temp_files = []


def create_default_sets() -> List[MacroPadSetInfo]:
    """Create 2 default macropad sets."""
    return [
        MacroPadSetInfo(
            name="Editing",
            set_marker_id=12,
            button_labels=["Copy", "Paste", "Undo", "Redo", "Cut", "Save", "Select All", "Delete", "Find", "Replace", "New", "Open"]
        ),
        MacroPadSetInfo(
            name="Media",
            set_marker_id=13,
            button_labels=["Play/Pause", "Stop", "Previous", "Next", "Vol -", "Vol +", "Mute", "Screenshot", "Record", "Fullscreen", "Rewind", "Forward"]
        ),
    ]


def create_sets_from_settings(setting) -> List[MacroPadSetInfo]:
    """
    Create MacroPadSetInfo list from user's settings.

    Args:
        setting: Setting object containing macropad_sets

    Returns:
        List of MacroPadSetInfo with labels from user configuration
    """
    sets = []
    for macro_set in setting.macropad_sets:
        # Extract labels from buttons (indices 0-11)
        labels = []
        for i in range(12):
            btn = macro_set.buttons.get(i)
            if btn and btn.label:
                labels.append(btn.label)
            else:
                labels.append(f"Button {i + 1}")

        sets.append(MacroPadSetInfo(
            name=macro_set.name,
            set_marker_id=macro_set.set_marker_id,
            button_labels=labels
        ))

    # If no sets configured, use defaults
    if not sets:
        return create_default_sets()

    return sets


