# -*- coding: utf-8 -*-
"""
identity_first_particle_tracking_sum_subtracted_support_v3_ready.py

Identity-first particle tracking pipeline.

Main logic
----------
1. Detect particles on the original real frames.
2. Build candidate SUM projection immobile regions.
3. Keep only the brightest core pixels inside each region (top fraction).
4. Subtract that SUM-core mask from every frame BEFORE building rolling MAX support.
5. Build rolling MAX support from the cleaned support stack.
6. Validate which SUM-core regions are true anchors using original detections.
7. Track particles with persistent global IDs.
8. Reuse the same anchor ID when an anchor reappears near its home.
9. Use one central colour mapping: track_id -> RGB, reused everywhere.
"""

import json
import math
import os
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology
from skimage.color import gray2rgb
from skimage.draw import line
from skimage.segmentation import find_boundaries, watershed


# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

UNTRACKED_COLOR = (255, 255, 255)

PIXEL_SIZE_NM = 110.0
FRAME_INTERVAL_S = 0.05
PIXEL_SIZE_UM = PIXEL_SIZE_NM / 1000.0

LAST_SETTINGS_FILENAME = "last_used_tracking_settings.json"


# =============================================================================
# GLOBAL TRACK COLOR MANAGER
# =============================================================================

class TrackColorManager:
    _color_map = {}

    @classmethod
    def reset(cls):
        cls._color_map = {}

    @classmethod
    def get_color(cls, track_id):
        if track_id is None:
            return UNTRACKED_COLOR

        track_id = int(track_id)

        if track_id not in cls._color_map:
            rng = np.random.default_rng(seed=track_id)
            color = tuple(int(v) for v in rng.integers(50, 255, size=3))
            cls._color_map[track_id] = color

        return cls._color_map[track_id]

    @classmethod
    def build_color_map_from_tracks(cls, track_ids):
        for tid in track_ids:
            cls.get_color(int(tid))

    @classmethod
    def get_full_map(cls):
        return cls._color_map.copy()


# =============================================================================
# SETTINGS STORAGE
# =============================================================================

class SettingsStore:
    @staticmethod
    def get_settings_path():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), LAST_SETTINGS_FILENAME)

    @classmethod
    def load_last_settings(cls):
        path = cls.get_settings_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @classmethod
    def save_last_settings(cls, settings):
        path = cls.get_settings_path()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)


# =============================================================================
# FILE SELECTION
# =============================================================================

class FileSelector:
    @staticmethod
    def select_tracking_file():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Select tracking stack",
            "First select the TIFF stack used for detection and tracking."
        )
        path = filedialog.askopenfilename(
            title="Select TRACKING stack TIFF",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        root.destroy()
        return path

    @staticmethod
    def select_overlay_file():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Select overlay stack",
            "Now select the TIFF stack used only as background for overlay movies."
        )
        path = filedialog.askopenfilename(
            title="Select OVERLAY stack TIFF",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        root.destroy()
        return path

    @staticmethod
    def select_output_folder():
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select output folder")
        root.destroy()
        return folder


# =============================================================================
# SETTINGS DIALOG
# =============================================================================

class SettingsDialog:
    DEFAULTS = {
        "min_area": 3,
        "max_area": 10000,
        "use_touching_split": True,
        "touch_split_min_peak_distance": 2,

        "rolling_projection_window_size": 3,
        "projection_corridor_radius": 1,
        "projection_support_threshold": 0.25,

        "max_assignment_distance": 7.0,
        "max_reclaim_distance": 8.0,
        "max_missed_frames": 3,

        "distance_weight": 0.34,
        "area_weight": 0.14,
        "mask_overlap_weight": 0.24,
        "motion_weight": 0.10,
        "projection_support_weight": 0.18,

        "min_assignment_score": 0.28,
        "min_reclaim_score": 0.22,

        "min_track_length": 3,
        "max_msd_lag": 10,

        "sum_region_min_area": 2,
        "sum_region_max_area": 20000,
        "sum_region_dilation_radius": 1,
        "sum_region_threshold_percentile": 85.0,
        "sum_core_keep_fraction": 0.10,

        "save_sum_masked_anchor_movie": True,
        "use_anchor_validation": True,
        "anchor_detection_radius": 5.0,
        "anchor_min_presence_fraction": 0.30,
        "anchor_max_jitter_pixels": 3.5,
        "anchor_min_frames_present": 4,
        "anchor_max_allowed_gap": 5,

        "use_anchor_calibration": True,
        "anchor_calibration_strength": 1.0,

        "anchor_home_reclaim_radius": 4.0,
        "anchor_extra_missed_frames": 12,
        "anchor_reclaim_area_tolerance": 0.70,

        "persistent_top_n_moving": 3,
        "persistent_top_n_immobile": 3,

        "save_sum_projection_region_movie": True,
        "save_cleaned_support_stack_movie": False,
        "save_rolling_projection_movie": True,
        "save_support_movie": True,
    }

    @classmethod
    def _load_starting_defaults(cls):
        merged = cls.DEFAULTS.copy()
        previous = SettingsStore.load_last_settings()
        if isinstance(previous, dict):
            for key, value in previous.items():
                if key in merged:
                    merged[key] = value
        return merged

    @classmethod
    def ask(cls):
        result = cls._load_starting_defaults()

        root = tk.Tk()
        root.title("Identity-first tracking with SUM-subtracted support")
        root.geometry("1380x980")
        root.minsize(1040, 760)

        entries = {}
        submitted = {"ok": False}

        outer = tk.Frame(root)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        v_scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        scrollable = tk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        scrollable.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.delta != 0:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        container = tk.Frame(scrollable, padx=14, pady=14)
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Identity-first tracking with SUM-subtracted support",
            font=("Arial", 15, "bold"),
            anchor="w"
        ).pack(fill="x", pady=(0, 12))

        tk.Label(
            container,
            text=(
                "Current logic:\n"
                "- SUM projection finds immobile regions first\n"
                "- only the brightest core pixels inside each SUM region are kept\n"
                "- those SUM-core pixels are subtracted from frames BEFORE rolling MAX support is built\n"
                "- real detections still use the original stack\n"
                "- validated anchors can reclaim the same ID near their home position\n"
                "- one central colour mapping is reused in every movie"
            ),
            justify="left",
            anchor="w",
            font=("Arial", 10),
            bg="#eef5ff",
            relief="groove",
            padx=10,
            pady=8
        ).pack(fill="x", pady=(0, 12))

        content = tk.Frame(container)
        content.pack(fill="both", expand=True)

        columns = []
        for _ in range(4):
            col = tk.Frame(content)
            col.pack(side="left", fill="both", expand=True, padx=(0, 18), anchor="n")
            columns.append(col)

        def add_row(parent, row_idx, label_text, key):
            tk.Label(
                parent,
                text=label_text,
                anchor="w",
                justify="left",
                font=("Arial", 11),
                wraplength=280
            ).grid(row=row_idx, column=0, padx=8, pady=8, sticky="nw")

            entry = tk.Entry(parent, width=18, font=("Arial", 11))
            entry.insert(0, str(result[key]))
            entry.grid(row=row_idx, column=1, padx=8, pady=8, sticky="w")
            entries[key] = entry

        blocks = [
            ("Detection / tracking", [
                ("Minimum area", "min_area"),
                ("Maximum area", "max_area"),
                ("Touch split minimum peak distance", "touch_split_min_peak_distance"),
                ("Rolling projection window size", "rolling_projection_window_size"),
                ("Projection corridor radius", "projection_corridor_radius"),
                ("Projection support threshold", "projection_support_threshold"),
                ("Maximum assignment distance", "max_assignment_distance"),
                ("Maximum reclaim distance", "max_reclaim_distance"),
                ("Maximum missed frames", "max_missed_frames"),
                ("Minimum assignment score", "min_assignment_score"),
                ("Minimum reclaim score", "min_reclaim_score"),
                ("Minimum track length", "min_track_length"),
                ("Maximum MSD lag", "max_msd_lag"),
            ]),
            ("Scoring weights", [
                ("Distance weight", "distance_weight"),
                ("Area weight", "area_weight"),
                ("Mask overlap weight", "mask_overlap_weight"),
                ("Motion weight", "motion_weight"),
                ("Projection support weight", "projection_support_weight"),
            ]),
            ("SUM immobile regions / anchors", [
                ("SUM region minimum area", "sum_region_min_area"),
                ("SUM region maximum area", "sum_region_max_area"),
                ("SUM region dilation radius", "sum_region_dilation_radius"),
                ("SUM threshold percentile", "sum_region_threshold_percentile"),
                ("SUM core keep fraction", "sum_core_keep_fraction"),
                ("Anchor detection radius", "anchor_detection_radius"),
                ("Anchor minimum presence fraction", "anchor_min_presence_fraction"),
                ("Anchor maximum jitter (pixels)", "anchor_max_jitter_pixels"),
                ("Anchor minimum frames present", "anchor_min_frames_present"),
                ("Anchor maximum allowed gap", "anchor_max_allowed_gap"),
            ]),
            ("Anchor reclaim / outputs", [
                ("Anchor calibration strength", "anchor_calibration_strength"),
                ("Anchor home reclaim radius", "anchor_home_reclaim_radius"),
                ("Anchor extra missed frames", "anchor_extra_missed_frames"),
                ("Anchor reclaim area tolerance", "anchor_reclaim_area_tolerance"),
                ("Persistent movie: top N moving", "persistent_top_n_moving"),
                ("Persistent movie: top N immobile", "persistent_top_n_immobile"),
            ]),
        ]

        for col, (title, items) in zip(columns, blocks):
            tk.Label(col, text=title, font=("Arial", 12, "bold"), anchor="w").grid(
                row=0, column=0, padx=8, pady=(0, 10), sticky="w"
            )
            for idx, (label_text, key) in enumerate(items, start=1):
                add_row(col, idx, label_text, key)

        check_parent = tk.Frame(container)
        check_parent.pack(fill="x", pady=(16, 12))

        use_touching_split_var = tk.BooleanVar(value=result["use_touching_split"])
        use_anchor_validation_var = tk.BooleanVar(value=result["use_anchor_validation"])
        use_anchor_calibration_var = tk.BooleanVar(value=result["use_anchor_calibration"])
        save_sum_regions_var = tk.BooleanVar(value=result["save_sum_projection_region_movie"])
        save_cleaned_support_stack_var = tk.BooleanVar(value=result["save_cleaned_support_stack_movie"])
        save_rolling_var = tk.BooleanVar(value=result["save_rolling_projection_movie"])
        save_support_var = tk.BooleanVar(value=result["save_support_movie"])

        for text, var in [
            ("Use touching split", use_touching_split_var),
            ("Use anchor validation", use_anchor_validation_var),
            ("Use anchor calibration", use_anchor_calibration_var),
            ("Save SUM immobile regions image", save_sum_regions_var),
            ("Save cleaned support stack movie", save_cleaned_support_stack_var),
            ("Save rolling projection movie", save_rolling_var),
            ("Save support movie", save_support_var),
        ]:
            tk.Checkbutton(
                check_parent,
                text=text,
                variable=var,
                font=("Arial", 11)
            ).pack(anchor="w")

        def on_ok():
            try:
                int_keys = [
                    "min_area", "max_area", "touch_split_min_peak_distance",
                    "rolling_projection_window_size", "projection_corridor_radius",
                    "max_missed_frames", "min_track_length", "max_msd_lag",
                    "sum_region_min_area", "sum_region_max_area", "sum_region_dilation_radius",
                    "anchor_min_frames_present", "anchor_max_allowed_gap",
                    "anchor_extra_missed_frames", "persistent_top_n_moving",
                    "persistent_top_n_immobile",
                ]
                float_keys = [
                    "projection_support_threshold", "max_assignment_distance",
                    "max_reclaim_distance", "distance_weight", "area_weight",
                    "mask_overlap_weight", "motion_weight", "projection_support_weight",
                    "min_assignment_score", "min_reclaim_score",
                    "sum_region_threshold_percentile", "sum_core_keep_fraction",
                    "anchor_detection_radius", "anchor_min_presence_fraction",
                    "anchor_max_jitter_pixels", "anchor_calibration_strength",
                    "anchor_home_reclaim_radius", "anchor_reclaim_area_tolerance",
                ]

                for key in int_keys:
                    result[key] = int(entries[key].get())

                for key in float_keys:
                    result[key] = float(entries[key].get())

                result["use_touching_split"] = bool(use_touching_split_var.get())
                result["use_anchor_validation"] = bool(use_anchor_validation_var.get())
                result["use_anchor_calibration"] = bool(use_anchor_calibration_var.get())
                result["save_sum_projection_region_movie"] = bool(save_sum_regions_var.get())
                result["save_cleaned_support_stack_movie"] = bool(save_cleaned_support_stack_var.get())
                result["save_rolling_projection_movie"] = bool(save_rolling_var.get())
                result["save_support_movie"] = bool(save_support_var.get())

                submitted["ok"] = True
                root.destroy()

            except Exception as error:
                messagebox.showerror("Invalid input", str(error))

        button_frame = tk.Frame(container)
        button_frame.pack(fill="x", pady=(0, 6))

        tk.Button(
            button_frame,
            text="Start tracking",
            width=22,
            height=2,
            command=on_ok,
            font=("Arial", 11, "bold")
        ).pack(side="left", padx=12)

        tk.Button(
            button_frame,
            text="Cancel",
            width=16,
            height=2,
            command=root.destroy,
            font=("Arial", 11)
        ).pack(side="left", padx=12)

        root.mainloop()

        try:
            canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass

        if not submitted["ok"]:
            return None

        SettingsStore.save_last_settings(result)
        return result


# =============================================================================
# PROGRESS WINDOW
# =============================================================================

class ProgressWindow:
    def __init__(self, title="Processing", maximum=100):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("980x220")
        self.root.minsize(900, 200)

        self.status_var = tk.StringVar(value="Starting...")
        self.percent_var = tk.StringVar(value="0%")

        tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            wraplength=920,
            font=("Arial", 11)
        ).pack(fill="x", padx=16, pady=(16, 8))

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=900,
            mode="determinate",
            maximum=maximum
        )
        self.progress.pack(fill="x", padx=16, pady=6)

        tk.Label(
            self.root,
            textvariable=self.percent_var,
            anchor="e",
            font=("Arial", 11, "bold")
        ).pack(fill="x", padx=16, pady=(4, 12))

        self.root.update_idletasks()
        self.root.update()

    def set_progress(self, value, maximum=None, text=None):
        if maximum is not None:
            self.progress["maximum"] = maximum

        self.progress["value"] = value
        max_value = float(self.progress["maximum"])
        percent = int(round((float(value) / max_value) * 100)) if max_value > 0 else 0
        self.percent_var.set(f"{percent}%")

        if text is not None:
            self.status_var.set(text)

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        try:
            self.root.update_idletasks()
            self.root.destroy()
        except Exception:
            pass


# =============================================================================
# STACK LOADING
# =============================================================================

class StackLoader:
    @staticmethod
    def load_tiff_stack(file_path):
        stack = tiff.imread(file_path)
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D TIFF stack (T, Y, X), got shape {stack.shape}")
        return stack

    @staticmethod
    def validate_same_shape(stack_a, stack_b):
        if stack_a.shape != stack_b.shape:
            raise ValueError(f"Tracking stack shape {stack_a.shape} does not match overlay stack shape {stack_b.shape}")


# =============================================================================
# SPLITTING / DETECTION
# =============================================================================

class TouchingObjectSplitter:
    def __init__(self, min_area, max_area, use_touching_split=True, min_peak_distance=2):
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.use_touching_split = bool(use_touching_split)
        self.min_peak_distance = int(min_peak_distance)

    @staticmethod
    def _is_binary_like(mask_frame):
        values = np.unique(mask_frame)
        if values.size <= 2:
            return True
        if values.size <= 3 and np.all(np.isin(values, [0, 1, 255])):
            return True
        return False

    def binarize_frame(self, frame):
        arr = np.asarray(frame)

        if arr.dtype == bool:
            return arr.copy()

        if self._is_binary_like(arr):
            return arr > 0

        arr = arr.astype(np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=bool)

        valid = arr[finite]
        if np.max(valid) <= np.min(valid):
            return arr > 0

        try:
            thresh = filters.threshold_otsu(valid)
            binary = arr > thresh
        except Exception:
            binary = arr > 0

        return binary

    def _split_region_with_watershed(self, raw_frame, labelled, region):
        region_mask = (labelled == int(region.label))
        if not np.any(region_mask):
            return [(region_mask, region)]

        coords = np.argwhere(region_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        local_mask = region_mask[y0:y1, x0:x1]
        local_raw = raw_frame[y0:y1, x0:x1].astype(np.float32)

        masked_vals = local_raw[local_mask]
        if masked_vals.size == 0 or np.max(masked_vals) <= np.min(masked_vals):
            return [(region_mask, region)]

        raw_in_mask = local_raw.copy()
        raw_in_mask[~local_mask] = 0.0

        peaks = feature.peak_local_max(
            raw_in_mask,
            min_distance=self.min_peak_distance,
            labels=local_mask.astype(np.uint8),
            num_peaks=8,
            exclude_border=False
        )

        if peaks.shape[0] < 2:
            return [(region_mask, region)]

        markers = np.zeros_like(local_mask, dtype=np.int32)
        for idx, (py, px) in enumerate(peaks, start=1):
            markers[int(py), int(px)] = idx

        markers = ndi.label(markers > 0)[0]
        if np.max(markers) < 2:
            return [(region_mask, region)]

        distance_map = ndi.distance_transform_edt(local_mask)
        ws = watershed(-distance_map, markers=markers, mask=local_mask)

        pieces = []
        for sub_id in np.unique(ws):
            if sub_id == 0:
                continue

            sub_local = (ws == sub_id)
            if np.sum(sub_local) < self.min_area:
                continue

            full_mask = np.zeros_like(region_mask, dtype=bool)
            full_mask[y0:y1, x0:x1] = sub_local

            temp_label = measure.label(full_mask)
            props = measure.regionprops(temp_label, intensity_image=raw_frame)
            if len(props) == 0:
                continue

            piece_region = props[0]
            if piece_region.area < self.min_area or piece_region.area > self.max_area:
                continue

            pieces.append((full_mask, piece_region))

        if len(pieces) < 2:
            return [(region_mask, region)]

        return pieces

    def split_frame(self, raw_frame, mask_frame):
        binary = self.binarize_frame(mask_frame)
        binary = morphology.remove_small_objects(binary, min_size=self.min_area)
        labelled = measure.label(binary)
        props = measure.regionprops(labelled, intensity_image=raw_frame)

        if len(props) == 0:
            return labelled.astype(np.int32)

        new_label = np.zeros_like(labelled, dtype=np.int32)
        next_id = 1

        for region in props:
            if region.area < self.min_area or region.area > self.max_area:
                continue

            if not self.use_touching_split:
                new_label[labelled == region.label] = next_id
                next_id += 1
                continue

            if region.area >= max(self.min_area * 2, 6):
                pieces = self._split_region_with_watershed(raw_frame, labelled, region)
            else:
                pieces = [((labelled == region.label), region)]

            for piece_mask, piece_region in pieces:
                if piece_region.area < self.min_area or piece_region.area > self.max_area:
                    continue
                new_label[piece_mask] = next_id
                next_id += 1

        return new_label.astype(np.int32)


class DetectionExtractor:
    def __init__(self, min_area, max_area, use_touching_split=True, touch_split_min_peak_distance=2):
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.splitter = TouchingObjectSplitter(
            min_area=min_area,
            max_area=max_area,
            use_touching_split=use_touching_split,
            min_peak_distance=touch_split_min_peak_distance,
        )

    def _make_detection_dict(self, frame_index, region, raw_frame):
        coords = region.coords
        y, x = region.centroid
        bbox = tuple(int(v) for v in region.bbox)
        intensity_values = raw_frame[coords[:, 0], coords[:, 1]].astype(np.float32)

        return {
            "frame": int(frame_index),
            "label_id": int(region.label),
            "y": float(y),
            "x": float(x),
            "area": float(region.area),
            "bbox_y0": int(bbox[0]),
            "bbox_x0": int(bbox[1]),
            "bbox_y1": int(bbox[2]),
            "bbox_x1": int(bbox[3]),
            "mean_intensity": float(np.mean(intensity_values)) if intensity_values.size > 0 else 0.0,
            "max_intensity": float(np.max(intensity_values)) if intensity_values.size > 0 else 0.0,
            "eccentricity": float(getattr(region, "eccentricity", 0.0)),
            "coords": coords.copy(),
        }

    def detect_in_frame(self, raw_stack, mask_stack, frame_index):
        raw_frame = raw_stack[frame_index]
        labelled = self.splitter.split_frame(raw_frame=raw_frame, mask_frame=mask_stack[frame_index])

        props = measure.regionprops(labelled, intensity_image=raw_frame)
        rows = []

        for region in props:
            if region.area < self.min_area or region.area > self.max_area:
                continue
            rows.append(self._make_detection_dict(frame_index, region, raw_frame))

        return rows, labelled

    def detect_in_stack(self, raw_stack, mask_stack, progress_window=None, progress_start=0, progress_end=20):
        all_rows = []
        label_maps = []
        detection_maps = []
        n_frames = mask_stack.shape[0]

        for frame_index in range(n_frames):
            rows, labelled = self.detect_in_frame(raw_stack, mask_stack, frame_index)
            all_rows.extend(rows)
            label_maps.append(labelled.astype(np.int32))
            detection_maps.append(rows)

            if progress_window is not None:
                progress_value = progress_start + ((frame_index + 1) / n_frames) * (progress_end - progress_start)
                progress_window.set_progress(
                    progress_value,
                    text=f"Detecting particles... frame {frame_index + 1}/{n_frames}"
                )

        label_maps = np.stack(label_maps, axis=0)

        flat_rows = []
        for d in all_rows:
            row = d.copy()
            row.pop("coords", None)
            flat_rows.append(row)

        detections_df = pd.DataFrame(flat_rows) if len(flat_rows) > 0 else pd.DataFrame()

        return detections_df, label_maps, detection_maps


# =============================================================================
# SUPPORT STACK / PROJECTION BUILDING
# =============================================================================

class RollingProjectionBuilder:
    def __init__(self, window_size=3):
        self.window_size = int(window_size)

    def build_max_stack(self, stack):
        stack = np.asarray(stack)
        n_frames = stack.shape[0]

        if self.window_size < 1 or self.window_size > n_frames:
            raise ValueError("Invalid rolling projection window size")

        projected_frames = []
        window_ranges = []

        for start in range(0, n_frames - self.window_size + 1):
            end = start + self.window_size
            projected_frames.append(np.max(stack[start:end], axis=0))
            window_ranges.append((start, end - 1))

        return np.stack(projected_frames, axis=0), window_ranges


class ProjectionSupportBuilder:
    def __init__(self, splitter, min_area, window_size, support_threshold=0.25, corridor_radius=1):
        self.splitter = splitter
        self.min_area = int(min_area)
        self.window_size = int(window_size)
        self.support_threshold = float(support_threshold)
        self.corridor_radius = int(corridor_radius)

        self.builder = RollingProjectionBuilder(window_size=self.window_size)
        self.max_projection_stack = None
        self.window_ranges = []
        self.support_masks = None

    def build(self, cleaned_support_stack):
        self.max_projection_stack, self.window_ranges = self.builder.build_max_stack(cleaned_support_stack)

        masks = []
        for i in range(self.max_projection_stack.shape[0]):
            binary = self.splitter.binarize_frame(self.max_projection_stack[i])
            binary = morphology.remove_small_objects(binary, min_size=self.min_area)
            binary = morphology.binary_closing(binary, morphology.disk(1))
            masks.append(binary.astype(bool))

        self.support_masks = np.stack(masks, axis=0)

    def is_ready(self):
        return self.max_projection_stack is not None and self.support_masks is not None and len(self.window_ranges) > 0

    def windows_for_frame(self, frame_idx):
        relevant = []
        for win_idx, (start, end) in enumerate(self.window_ranges):
            if start <= frame_idx <= end:
                relevant.append(win_idx)
        return relevant

    def bridging_windows(self, frame_a, frame_b):
        lo = min(int(frame_a), int(frame_b))
        hi = max(int(frame_a), int(frame_b))
        relevant = []

        for win_idx, (start, end) in enumerate(self.window_ranges):
            if start <= lo <= end:
                relevant.append(win_idx)
            elif start <= hi <= end:
                relevant.append(win_idx)
            elif lo <= end and hi >= start:
                relevant.append(win_idx)

        return sorted(set(relevant))

    @staticmethod
    def _patch_hit(mask, y, x, radius):
        y = int(round(y))
        x = int(round(x))
        h, w = mask.shape
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        return bool(np.any(mask[y0:y1, x0:x1]))

    def point_support_fraction(self, frame_idx, y, x):
        if not self.is_ready():
            return 0.0

        windows = self.windows_for_frame(frame_idx)
        if len(windows) == 0:
            return 0.0

        hits = 0
        for win_idx in windows:
            if self._patch_hit(self.support_masks[win_idx], y, x, self.corridor_radius):
                hits += 1

        return float(hits) / float(len(windows))

    def line_support_fraction(self, frame_a, y0, x0, frame_b, y1, x1):
        if not self.is_ready():
            return 0.0

        windows = self.bridging_windows(frame_a, frame_b)
        if len(windows) == 0:
            return 0.0

        rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
        fractions = []

        for win_idx in windows:
            mask = self.support_masks[win_idx]
            h, w = mask.shape
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr_valid = rr[valid]
            cc_valid = cc[valid]

            if len(rr_valid) == 0:
                fractions.append(0.0)
                continue

            hits = 0
            total = 0
            for r, c in zip(rr_valid, cc_valid):
                total += 1
                if self._patch_hit(mask, r, c, self.corridor_radius):
                    hits += 1

            fractions.append(float(hits) / float(total) if total > 0 else 0.0)

        return float(np.mean(fractions)) if len(fractions) > 0 else 0.0

# =============================================================================
# SUM IMMOBILE REGIONS
# =============================================================================

class SumProjectionImmobileRegionBuilder:
    def __init__(self, min_area, max_area, dilation_radius=1, threshold_percentile=85.0, keep_top_fraction=0.10):
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.dilation_radius = int(dilation_radius)
        self.threshold_percentile = float(threshold_percentile)
        self.keep_top_fraction = float(keep_top_fraction)

    @staticmethod
    def _top_fraction_mask_within_region(sum_projection, region_mask, keep_top_fraction):
        coords = np.argwhere(region_mask)
        if coords.size == 0:
            return np.zeros_like(region_mask, dtype=bool)

        values = sum_projection[region_mask].astype(np.float32)
        n_pixels = len(values)
        n_keep = max(1, int(np.ceil(n_pixels * float(keep_top_fraction))))

        order = np.argsort(values)[::-1]
        keep_idx = order[:n_keep]

        out = np.zeros_like(region_mask, dtype=bool)
        keep_coords = coords[keep_idx]
        out[keep_coords[:, 0], keep_coords[:, 1]] = True
        return out

    def build(self, tracking_stack):
        sum_projection = np.sum(np.asarray(tracking_stack, dtype=np.float32), axis=0)

        finite = np.isfinite(sum_projection)
        if not np.any(finite):
            empty_df = pd.DataFrame(columns=[
                "sum_region_id", "sum_region_y", "sum_region_x",
                "sum_region_area", "core_y", "core_x",
                "core_area", "core_fraction_kept"
            ])
            return (
                sum_projection,
                np.zeros(sum_projection.shape, dtype=np.int32),
                np.zeros(sum_projection.shape, dtype=bool),
                empty_df
            )

        vals = sum_projection[finite]
        threshold = np.percentile(vals, self.threshold_percentile)
        binary = sum_projection >= threshold

        binary = morphology.remove_small_objects(binary, min_size=self.min_area)
        binary = morphology.binary_closing(binary, morphology.disk(1))

        labelled = measure.label(binary)
        props = measure.regionprops(labelled, intensity_image=sum_projection)

        selected_mask = np.zeros(sum_projection.shape, dtype=bool)
        rows = []

        for region in props:
            if region.area < self.min_area or region.area > self.max_area:
                continue

            region_mask = (labelled == int(region.label))
            core_mask = self._top_fraction_mask_within_region(
                sum_projection=sum_projection,
                region_mask=region_mask,
                keep_top_fraction=self.keep_top_fraction
            )

            if self.dilation_radius > 0:
                core_mask = morphology.binary_dilation(
                    core_mask,
                    morphology.disk(self.dilation_radius)
                )

            core_area = int(np.sum(core_mask))
            if core_area < 1:
                continue

            selected_mask |= core_mask

            core_coords = np.argwhere(core_mask)
            core_y = float(np.mean(core_coords[:, 0]))
            core_x = float(np.mean(core_coords[:, 1]))

            rows.append({
                "sum_region_id": int(region.label),
                "sum_region_y": float(region.centroid[0]),
                "sum_region_x": float(region.centroid[1]),
                "sum_region_area": float(region.area),
                "core_y": core_y,
                "core_x": core_x,
                "core_area": core_area,
                "core_fraction_kept": float(core_area) / float(region.area) if region.area > 0 else np.nan,
            })

        regions_df = pd.DataFrame(rows) if len(rows) > 0 else pd.DataFrame(
            columns=[
                "sum_region_id", "sum_region_y", "sum_region_x",
                "sum_region_area", "core_y", "core_x",
                "core_area", "core_fraction_kept"
            ]
        )

        return sum_projection, labelled, selected_mask.astype(bool), regions_df


# =============================================================================
# ANCHOR VALIDATION
# =============================================================================

class AnchorValidator:
    def __init__(
        self,
        detection_radius,
        min_presence_fraction,
        max_jitter_pixels,
        min_frames_present,
        max_allowed_gap
    ):
        self.detection_radius = float(detection_radius)
        self.min_presence_fraction = float(min_presence_fraction)
        self.max_jitter_pixels = float(max_jitter_pixels)
        self.min_frames_present = int(min_frames_present)
        self.max_allowed_gap = int(max_allowed_gap)

    @staticmethod
    def _distance(y0, x0, y1, x1):
        return float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

    @staticmethod
    def _longest_missing_gap(present_flags):
        longest = 0
        current = 0
        for flag in present_flags:
            if flag:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return int(longest)

    def validate(self, regions_df, detection_maps, n_frames):
        if regions_df is None or len(regions_df) == 0:
            return pd.DataFrame()

        rows = []

        for _, region in regions_df.iterrows():
            cy = float(region["core_y"]) if "core_y" in region and pd.notna(region["core_y"]) else float(region["sum_region_y"])
            cx = float(region["core_x"]) if "core_x" in region and pd.notna(region["core_x"]) else float(region["sum_region_x"])

            rows.append({
                "sum_region_id": int(region["sum_region_id"]),
                "anchor_y": cy,
                "anchor_x": cx,
                "sum_region_area": float(region.get("sum_region_area", np.nan)),
                "core_area": float(region.get("core_area", np.nan)),
                "core_fraction_kept": float(region.get("core_fraction_kept", np.nan)),
                "times_detected_in_region": int(n_frames),
                "times_not_detected_in_region": 0,
                "presence_fraction": 1.0,
                "n_frames_present": int(n_frames),
                "n_frames_missing": 0,
                "longest_missing_gap": 0,
                "mean_detected_y": cy,
                "mean_detected_x": cx,
                "mean_jitter_pixels": 0.0,
                "max_jitter_pixels": 0.0,
                "area_mean": np.nan,
                "area_std": np.nan,
                "area_cv": np.nan,
                "mean_distance_to_core": 0.0,
                "mean_detected_mean_intensity": np.nan,
                "mean_detected_max_intensity": np.nan,
                "first_present_frame": 0,
                "last_present_frame": int(n_frames - 1),
                "is_anchor": True,
                })

        return pd.DataFrame(rows)

# =============================================================================
# AUTO CALIBRATION
# =============================================================================

class AutoParameterCalibrator:
    def __init__(self, calibration_strength=1.0):
        self.calibration_strength = float(calibration_strength)

    @staticmethod
    def _safe_percentile(values, q, fallback):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float(fallback)
        return float(np.percentile(values, q))

    def calibrate(self, settings, anchor_df):
        settings_used = settings.copy()

        if anchor_df is None or len(anchor_df) == 0:
            return settings_used, pd.DataFrame([{
                "parameter": "anchor_calibration",
                "old_value": "not_run",
                "new_value": "not_run",
                "reason": "no anchor data"
            }])

        good = anchor_df.copy()
        if len(good) == 0:
            return settings_used, pd.DataFrame([{
                "parameter": "anchor_calibration",
                "old_value": "no_good_anchors",
                "new_value": "no_good_anchors",
                "reason": "no anchors passed criteria"
            }])

        jitter95 = self._safe_percentile(good["max_jitter_pixels"], 95, settings["max_assignment_distance"])
        gap95 = self._safe_percentile(good["longest_missing_gap"], 95, settings["max_missed_frames"])
        area_cv95 = self._safe_percentile(good["area_cv"], 95, 0.25)

        strength = max(0.0, self.calibration_strength)

        old_assignment = float(settings_used["max_assignment_distance"])
        old_reclaim = float(settings_used["max_reclaim_distance"])
        old_missed = int(settings_used["max_missed_frames"])
        old_area_weight = float(settings_used["area_weight"])
        old_mask_overlap_weight = float(settings_used["mask_overlap_weight"])

        suggested_assignment = max(old_assignment, jitter95 + 1.0)
        suggested_reclaim = max(old_reclaim, jitter95 + 2.0)
        suggested_missed = max(old_missed, int(math.ceil(gap95)))
        suggested_area_weight = old_area_weight
        suggested_mask_overlap_weight = old_mask_overlap_weight

        if np.isfinite(area_cv95):
            if area_cv95 > 0.40:
                suggested_area_weight = max(0.05, old_area_weight * 0.8)
                suggested_mask_overlap_weight = min(0.40, old_mask_overlap_weight * 1.1)
            elif area_cv95 < 0.15:
                suggested_area_weight = min(0.35, old_area_weight * 1.1)

        def blend(old_val, new_val):
            return old_val + strength * (new_val - old_val)

        settings_used["max_assignment_distance"] = float(blend(old_assignment, suggested_assignment))
        settings_used["max_reclaim_distance"] = float(blend(old_reclaim, suggested_reclaim))
        settings_used["max_missed_frames"] = int(round(blend(old_missed, suggested_missed)))
        settings_used["area_weight"] = float(blend(old_area_weight, suggested_area_weight))
        settings_used["mask_overlap_weight"] = float(blend(old_mask_overlap_weight, suggested_mask_overlap_weight))

        rows = [
            {
                "parameter": "max_assignment_distance",
                "old_value": old_assignment,
                "new_value": settings_used["max_assignment_distance"],
                "reason": f"anchor jitter 95th percentile = {jitter95:.3f}"
            },
            {
                "parameter": "max_reclaim_distance",
                "old_value": old_reclaim,
                "new_value": settings_used["max_reclaim_distance"],
                "reason": f"anchor jitter 95th percentile = {jitter95:.3f}"
            },
            {
                "parameter": "max_missed_frames",
                "old_value": old_missed,
                "new_value": settings_used["max_missed_frames"],
                "reason": f"anchor missing-gap 95th percentile = {gap95:.3f}"
            },
            {
                "parameter": "area_weight",
                "old_value": old_area_weight,
                "new_value": settings_used["area_weight"],
                "reason": f"anchor area CV 95th percentile = {area_cv95:.3f}"
            },
            {
                "parameter": "mask_overlap_weight",
                "old_value": old_mask_overlap_weight,
                "new_value": settings_used["mask_overlap_weight"],
                "reason": f"anchor area CV 95th percentile = {area_cv95:.3f}"
            },
        ]

        total = (
            settings_used["distance_weight"] +
            settings_used["area_weight"] +
            settings_used["mask_overlap_weight"] +
            settings_used["motion_weight"] +
            settings_used["projection_support_weight"]
        )

        if total > 0:
            settings_used["distance_weight"] /= total
            settings_used["area_weight"] /= total
            settings_used["mask_overlap_weight"] /= total
            settings_used["motion_weight"] /= total
            settings_used["projection_support_weight"] /= total

        return settings_used, pd.DataFrame(rows)


# =============================================================================
# SIMPLE IMAGE RENDERERS
# =============================================================================

class SimpleImageRenderer:
    @staticmethod
    def normalize_frame(frame):
        frame = np.asarray(frame).astype(np.float32)
        finite = np.isfinite(frame)
        if not np.any(finite):
            return np.zeros(frame.shape, dtype=np.uint8)

        valid = frame[finite]
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            return np.zeros(frame.shape, dtype=np.uint8)

        scaled = (frame - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    @classmethod
    def sum_region_rgb(cls, projection_frame, region_label_map):
        gray = cls.normalize_frame(projection_frame)
        rgb = gray2rgb(gray)
        boundaries = find_boundaries(region_label_map > 0, mode="outer")
        rgb[boundaries] = np.array([255, 255, 255], dtype=np.uint8)
        return rgb.astype(np.uint8)

    @classmethod
    def support_movie_rgb(cls, projection_stack, support_masks):
        movie = []
        for i in range(projection_stack.shape[0]):
            gray = cls.normalize_frame(projection_stack[i])
            rgb = gray2rgb(gray)
            boundaries = find_boundaries(support_masks[i], mode="outer")
            rgb[boundaries] = np.array([255, 255, 255], dtype=np.uint8)
            movie.append(rgb.astype(np.uint8))
        return np.stack(movie, axis=0)


# =============================================================================
# OVERLAY RENDERER
# =============================================================================

class OverlayRenderer:
    def __init__(self, show_track_lines=True):
        self.show_track_lines = bool(show_track_lines)

    @staticmethod
    def _normalize_to_uint8(frame):
        frame = np.asarray(frame).astype(np.float32)
        finite = np.isfinite(frame)
        if not np.any(finite):
            return np.zeros(frame.shape, dtype=np.uint8)

        valid = frame[finite]
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            return np.zeros(frame.shape, dtype=np.uint8)

        scaled = (frame - vmin) / (vmax - vmin)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    @staticmethod
    def _draw_line(canvas, y0, x0, y1, x1, color):
        rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
        valid = (
            (rr >= 0) & (rr < canvas.shape[0]) &
            (cc >= 0) & (cc < canvas.shape[1])
        )
        canvas[rr[valid], cc[valid]] = color

    def draw_new_segments_only(
        self,
        line_canvas,
        frame_detections,
        previous_point_by_track,
        track_lengths,
        min_track_length_to_draw
    ):
        if not self.show_track_lines:
            return

        frame_detections = frame_detections.sort_values(["track_id", "frame"])

        for _, row in frame_detections.iterrows():
            track_id = int(row["track_id"])
            y = float(row["y"])
            x = float(row["x"])

            if int(track_lengths.get(track_id, 0)) < int(min_track_length_to_draw):
                previous_point_by_track[track_id] = (y, x)
                continue

            color = TrackColorManager.get_color(track_id)

            if track_id in previous_point_by_track:
                prev_y, prev_x = previous_point_by_track[track_id]
                self._draw_line(line_canvas, prev_y, prev_x, y, x, color)

            previous_point_by_track[track_id] = (y, x)

    def render_outline_and_lines_frame(self, raw_frame, label_map_frame, frame_detections, line_canvas):
        gray = self._normalize_to_uint8(raw_frame)
        rgb = gray2rgb(gray)

        boundaries = find_boundaries(label_map_frame > 0, mode="outer")
        rgb[boundaries] = np.array([255, 255, 255], dtype=np.uint8)

        if frame_detections is not None and len(frame_detections) > 0:
            for _, row in frame_detections.iterrows():
                track_id = int(row["track_id"]) if pd.notna(row["track_id"]) else None
                label_id = int(row["label_id"]) if pd.notna(row["label_id"]) else None
                color = TrackColorManager.get_color(track_id)

                if label_id is not None and label_id >= 0:
                    mask = (label_map_frame == label_id)
                    if np.any(mask):
                        region_boundary = find_boundaries(mask, mode="outer")
                        rgb[region_boundary] = np.array(color, dtype=np.uint8)

        if self.show_track_lines and line_canvas is not None:
            line_mask = np.any(line_canvas > 0, axis=2)
            rgb[line_mask] = line_canvas[line_mask]

        return rgb


# =============================================================================
# TRACKER DATA
# =============================================================================

@dataclass
class DetectionRecord:
    frame: int
    label_id: int
    local_detection_index: int
    y: float
    x: float
    area: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    mean_intensity: float
    max_intensity: float
    eccentricity: float
    coords: np.ndarray
    state_note: str = "visible"

    def bbox(self):
        return (self.bbox_y0, self.bbox_x0, self.bbox_y1, self.bbox_x1)


@dataclass
class ParticleIdentity:
    particle_id: int
    color: tuple
    history: list = field(default_factory=list)
    state: str = "active"
    missed_frames: int = 0
    created_frame: int = 0
    ended_frame: int = None
    is_anchor: bool = False
    anchor_key: int = None
    anchor_home_y: float = np.nan
    anchor_home_x: float = np.nan
    anchor_area_mean: float = np.nan

    def add_detection(self, detection: DetectionRecord, state_note="visible"):
        detection.state_note = state_note
        self.history.append(detection)
        self.missed_frames = 0
        self.state = "active"

    def mark_missed(self):
        self.missed_frames += 1
        self.state = "lost"

    def mark_ended(self, frame_idx):
        self.state = "ended"
        self.ended_frame = int(frame_idx)

    def last_detection(self):
        return self.history[-1] if len(self.history) > 0 else None

    def previous_detection(self):
        return self.history[-2] if len(self.history) > 1 else None

    def predicted_position(self):
        last_det = self.last_detection()
        if last_det is None:
            return None

        if self.is_anchor and np.isfinite(self.anchor_home_y) and np.isfinite(self.anchor_home_x):
            return (float(self.anchor_home_y), float(self.anchor_home_x))

        prev_det = self.previous_detection()
        if prev_det is None:
            return (float(last_det.y), float(last_det.x))

        dy = float(last_det.y - prev_det.y)
        dx = float(last_det.x - prev_det.x)
        return (float(last_det.y + dy), float(last_det.x + dx))

    def is_stationary_like(self, max_recent_disp=1.5, window=4):
        if self.is_anchor:
            return True

        if len(self.history) < 2:
            return False

        recent = self.history[-window:]
        if len(recent) < 2:
            return False

        disps = []
        for i in range(len(recent) - 1):
            dy = recent[i + 1].y - recent[i].y
            dx = recent[i + 1].x - recent[i].x
            disps.append(math.sqrt(dx * dx + dy * dy))

        return float(np.mean(disps)) <= float(max_recent_disp)


# =============================================================================
# IDENTITY TRACKER
# =============================================================================

class IdentityFirstTracker:
    def __init__(self, settings, projection_builder=None, anchor_df=None):
        self.settings = settings
        self.projection_builder = projection_builder
        self.anchor_df = anchor_df if anchor_df is not None else pd.DataFrame()

        self.max_assignment_distance = float(settings["max_assignment_distance"])
        self.max_reclaim_distance = float(settings["max_reclaim_distance"])
        self.max_missed_frames = int(settings["max_missed_frames"])

        self.distance_weight = float(settings["distance_weight"])
        self.area_weight = float(settings["area_weight"])
        self.mask_overlap_weight = float(settings["mask_overlap_weight"])
        self.motion_weight = float(settings["motion_weight"])
        self.projection_support_weight = float(settings["projection_support_weight"])

        self.min_assignment_score = float(settings["min_assignment_score"])
        self.min_reclaim_score = float(settings["min_reclaim_score"])
        self.projection_support_threshold = float(settings["projection_support_threshold"])

        self.anchor_home_reclaim_radius = float(settings["anchor_home_reclaim_radius"])
        self.anchor_extra_missed_frames = int(settings["anchor_extra_missed_frames"])
        self.anchor_reclaim_area_tolerance = float(settings["anchor_reclaim_area_tolerance"])

        self.particles = {}
        self.anchor_key_to_particle_id = {}
        self.next_particle_id = 0
        self.assignment_rows = []
        self.event_rows = []

        self.anchor_lookup = self._build_anchor_lookup(self.anchor_df)

    @staticmethod
    def _distance(y0, x0, y1, x1):
        return float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

    @staticmethod
    def _safe_similarity(a, b):
        a = float(a)
        b = float(b)
        denom = max(abs(a), abs(b), 1e-9)
        diff = abs(a - b) / denom
        return max(0.0, 1.0 - diff)

    @staticmethod
    def _bbox_iou(a, b):
        ay0, ax0, ay1, ax1 = a
        by0, bx0, by1, bx1 = b

        iy0 = max(ay0, by0)
        ix0 = max(ax0, bx0)
        iy1 = min(ay1, by1)
        ix1 = min(ax1, bx1)

        inter_h = max(0, iy1 - iy0)
        inter_w = max(0, ix1 - ix0)
        inter = inter_h * inter_w

        area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
        area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    @staticmethod
    def _coords_overlap_fraction(coords_a, coords_b):
        if coords_a is None or coords_b is None:
            return 0.0
        if len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0

        set_a = set((int(r), int(c)) for r, c in coords_a)
        set_b = set((int(r), int(c)) for r, c in coords_b)
        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    @staticmethod
    def _build_anchor_lookup(anchor_df):
        lookup = {}
        if anchor_df is None or len(anchor_df) == 0:
            return lookup

        good = anchor_df[anchor_df["is_anchor"] == True].copy()
        for _, row in good.iterrows():
            lookup[int(row["sum_region_id"])] = {
                "anchor_key": int(row["sum_region_id"]),
                "anchor_y": float(row["anchor_y"]),
                "anchor_x": float(row["anchor_x"]),
                "area_mean": float(row["area_mean"]) if pd.notna(row["area_mean"]) else np.nan,
                "times_detected_in_region": int(row["times_detected_in_region"]) if pd.notna(row.get("times_detected_in_region", np.nan)) else np.nan,
            }
        return lookup

    def _anchor_info_for_detection(self, detection: DetectionRecord):
        if len(self.anchor_lookup) == 0:
            return None

        best_info = None
        best_dist = None

        for _, info in self.anchor_lookup.items():
            dist = self._distance(info["anchor_y"], info["anchor_x"], detection.y, detection.x)
            if dist <= self.anchor_home_reclaim_radius:
                if best_info is None or dist < best_dist:
                    best_info = info
                    best_dist = dist

        return best_info

    def _create_or_reuse_anchor_particle(self, detection: DetectionRecord, anchor_info):
        anchor_key = int(anchor_info["anchor_key"])

        if anchor_key in self.anchor_key_to_particle_id:
            pid = int(self.anchor_key_to_particle_id[anchor_key])
            particle = self.particles[pid]
            particle.state = "active"
            particle.missed_frames = 0
            particle.add_detection(detection, state_note="reused_anchor_key")
            return particle, True

        pid = int(self.next_particle_id)
        self.next_particle_id += 1

        particle = ParticleIdentity(
            particle_id=pid,
            color=TrackColorManager.get_color(pid),
            history=[],
            state="active",
            missed_frames=0,
            created_frame=int(detection.frame),
            is_anchor=True,
            anchor_key=anchor_key,
            anchor_home_y=float(anchor_info["anchor_y"]),
            anchor_home_x=float(anchor_info["anchor_x"]),
            anchor_area_mean=float(anchor_info["area_mean"]) if anchor_info["area_mean"] is not None else np.nan,
        )
        particle.add_detection(detection, state_note="new_anchor")
        self.particles[pid] = particle
        self.anchor_key_to_particle_id[anchor_key] = pid

        self.event_rows.append({
            "frame": int(detection.frame),
            "particle_id": pid,
            "event_type": "created_anchor",
            "details": f"anchor key {anchor_key} created"
        })

        return particle, False

    def _create_particle(self, detection: DetectionRecord):
        pid = int(self.next_particle_id)
        self.next_particle_id += 1

        particle = ParticleIdentity(
            particle_id=pid,
            color=TrackColorManager.get_color(pid),
            history=[],
            state="active",
            missed_frames=0,
            created_frame=int(detection.frame),
            is_anchor=False,
        )
        particle.add_detection(detection, state_note="new_particle")
        self.particles[pid] = particle

        self.event_rows.append({
            "frame": int(detection.frame),
            "particle_id": pid,
            "event_type": "created",
            "details": "new particle created"
        })

        return particle
    
    def _candidate_projection_support(self, particle: ParticleIdentity, detection: DetectionRecord):
        if particle.is_anchor:
            return 0.0

        if self.projection_builder is None or not self.projection_builder.is_ready():
            return 0.0

        last_det = particle.last_detection()
        if last_det is None:
            return 0.0

        point_support = self.projection_builder.point_support_fraction(detection.frame, detection.y, detection.x)
        line_support = self.projection_builder.line_support_fraction(
            last_det.frame, last_det.y, last_det.x,
            detection.frame, detection.y, detection.x
        )
        return float((point_support + line_support) / 2.0)

    def _motion_similarity(self, particle: ParticleIdentity, detection: DetectionRecord):
        last_det = particle.last_detection()
        if last_det is None:
            return 0.0

        if particle.is_anchor:
            dist = self._distance(particle.anchor_home_y, particle.anchor_home_x, detection.y, detection.x)
            return max(0.0, 1.0 - (dist / max(self.anchor_home_reclaim_radius, 1e-9)))

        prev_det = particle.previous_detection()
        if prev_det is None:
            return 0.5

        if particle.is_stationary_like():
            dist = self._distance(last_det.y, last_det.x, detection.y, detection.x)
            return max(0.0, 1.0 - (dist / max(self.max_reclaim_distance, 1e-9)))

        pred_y, pred_x = particle.predicted_position()
        dist = self._distance(pred_y, pred_x, detection.y, detection.x)
        return max(0.0, 1.0 - (dist / max(self.max_reclaim_distance, 1e-9)))

    def _build_score(self, particle: ParticleIdentity, detection: DetectionRecord):
        last_det = particle.last_detection()
        if last_det is None:
            return None

        is_stationary = particle.is_stationary_like()

        if particle.is_anchor:
            pred_y = float(particle.anchor_home_y)
            pred_x = float(particle.anchor_home_x)
            endpoint_distance = self._distance(pred_y, pred_x, detection.y, detection.x)

            if endpoint_distance > self.anchor_home_reclaim_radius:
                return None

            if np.isfinite(particle.anchor_area_mean):
                area_score = self._safe_similarity(particle.anchor_area_mean, detection.area)
                if area_score < (1.0 - self.anchor_reclaim_area_tolerance):
                    return None
            else:
                area_score = self._safe_similarity(last_det.area, detection.area)

            bbox_overlap = self._bbox_iou(last_det.bbox(), detection.bbox())
            coords_overlap = self._coords_overlap_fraction(last_det.coords, detection.coords)
            mask_overlap_score = max(bbox_overlap, coords_overlap)

            distance_score = max(0.0, 1.0 - (endpoint_distance / max(self.anchor_home_reclaim_radius, 1e-9)))
            motion_score = self._motion_similarity(particle, detection)

            total_weight = self.distance_weight + self.area_weight + self.mask_overlap_weight + self.motion_weight
            final_score = (
                self.distance_weight * distance_score +
                self.area_weight * area_score +
                self.mask_overlap_weight * mask_overlap_score +
                self.motion_weight * motion_score
            ) / max(total_weight, 1e-9)

            if final_score < self.min_reclaim_score:
                return None

            return {
                "particle_id": int(particle.particle_id),
                "distance_score": float(distance_score),
                "area_score": float(area_score),
                "mask_overlap_score": float(mask_overlap_score),
                "motion_score": float(motion_score),
                "projection_support_score": 0.0,
                "final_score": float(final_score),
                "is_anchor": True,
                "is_stationary_like": True,
            }

        dist_limit = self.max_assignment_distance if particle.state == "active" else self.max_reclaim_distance
        min_score = self.min_assignment_score if particle.state == "active" else self.min_reclaim_score

        if is_stationary:
            pred_y = float(last_det.y)
            pred_x = float(last_det.x)
        else:
            pred_y, pred_x = particle.predicted_position()

        endpoint_distance = self._distance(pred_y, pred_x, detection.y, detection.x)
        if endpoint_distance > dist_limit:
            return None

        distance_score = max(0.0, 1.0 - (endpoint_distance / max(dist_limit, 1e-9)))
        area_score = self._safe_similarity(last_det.area, detection.area)

        bbox_overlap = self._bbox_iou(last_det.bbox(), detection.bbox())
        coords_overlap = self._coords_overlap_fraction(last_det.coords, detection.coords)
        mask_overlap_score = max(bbox_overlap, coords_overlap)

        motion_score = self._motion_similarity(particle, detection)
        projection_support = self._candidate_projection_support(particle, detection)

        if is_stationary:
            distance_score = min(1.0, distance_score + 0.10)
            mask_overlap_score = min(1.0, mask_overlap_score + 0.10)

        total_weight = (
            self.distance_weight +
            self.area_weight +
            self.mask_overlap_weight +
            self.motion_weight +
            self.projection_support_weight
        )

        final_score = (
            self.distance_weight * distance_score +
            self.area_weight * area_score +
            self.mask_overlap_weight * mask_overlap_score +
            self.motion_weight * motion_score +
            self.projection_support_weight * projection_support
        ) / max(total_weight, 1e-9)

        if final_score < min_score:
            return None

        if particle.state == "lost" and particle.missed_frames >= 2 and not is_stationary:
            if projection_support < self.projection_support_threshold:
                return None

        return {
            "particle_id": int(particle.particle_id),
            "distance_score": float(distance_score),
            "area_score": float(area_score),
            "mask_overlap_score": float(mask_overlap_score),
            "motion_score": float(motion_score),
            "projection_support_score": float(projection_support),
            "final_score": float(final_score),
            "is_anchor": False,
            "is_stationary_like": bool(is_stationary),
        }

    def _select_assignments(self, detections):
        candidate_rows = []

        candidate_particles = [
            p for p in self.particles.values()
            if p.state in ("active", "lost")
            ]

        for particle in candidate_particles:
            max_missed_allowed = self.max_missed_frames
            if particle.is_anchor:
                max_missed_allowed += self.anchor_extra_missed_frames

            if particle.state == "lost" and particle.missed_frames > max_missed_allowed:
                continue

            for det in detections:
                score_info = self._build_score(particle, det)
                
                if score_info is not None:
                    candidate_rows.append({
                        "particle_id": int(particle.particle_id),
                        "detection_local_id": int(det.local_detection_index),
                        "score": float(score_info["final_score"]),
                        "score_info": score_info,
                        "detection": det,
                        })

        candidate_rows.sort(
            key=lambda d: (
                d["score_info"]["is_anchor"],
                d["score"],
                d["score_info"]["mask_overlap_score"],
                d["score_info"]["distance_score"]
                ),
            reverse=True
            )

        assigned_particles = set()
        assigned_detections = set()
        accepted = []

        for cand in candidate_rows:
            pid = int(cand["particle_id"])
            did = int(cand["detection_local_id"])

            if pid in assigned_particles:
                continue

            if did in assigned_detections:
                continue

            accepted.append(cand)
            assigned_particles.add(pid)
            assigned_detections.add(did)

        return accepted

    def _assign_anchors_first(self, frame_idx, detections):
        accepted_detection_ids = set()
        accepted_particle_ids = set()

        if len(self.anchor_lookup) == 0:
            return accepted_detection_ids, accepted_particle_ids

        anchor_candidates = []

        for det in detections:
            for anchor_key, info in self.anchor_lookup.items():
                dist = self._distance(
                    info["anchor_y"],
                    info["anchor_x"],
                    det.y,
                    det.x
                )

                if dist <= self.anchor_home_reclaim_radius:
                    anchor_candidates.append({
                        "anchor_key": int(anchor_key),
                        "detection": det,
                        "distance": float(dist),
                        "anchor_info": info,
                    })

        anchor_candidates.sort(key=lambda row: row["distance"])

        used_anchor_keys = set()

        for cand in anchor_candidates:
            anchor_key = int(cand["anchor_key"])
            det = cand["detection"]

            if anchor_key in used_anchor_keys:
                continue

            if int(det.local_detection_index) in accepted_detection_ids:
                continue

            particle, reused_existing_anchor = self._create_or_reuse_anchor_particle(
                detection=det,
                anchor_info=cand["anchor_info"]
            )

            used_anchor_keys.add(anchor_key)
            accepted_detection_ids.add(int(det.local_detection_index))
            accepted_particle_ids.add(int(particle.particle_id))

            self.assignment_rows.append({
                "frame": int(det.frame),
                "label_id": int(det.label_id),
                "track_id": int(particle.particle_id),
                "particle_id": int(particle.particle_id),
                "y": float(det.y),
                "x": float(det.x),
                "area": float(det.area),
                "bbox_y0": int(det.bbox_y0),
                "bbox_x0": int(det.bbox_x0),
                "bbox_y1": int(det.bbox_y1),
                "bbox_x1": int(det.bbox_x1),
                "mean_intensity": float(det.mean_intensity),
                "max_intensity": float(det.max_intensity),
                "eccentricity": float(det.eccentricity),
                "assignment_state": "anchor_first_reused" if reused_existing_anchor else "anchor_first_created",
                "assignment_score": 1.0,
                "distance_score": np.nan,
                "area_score": np.nan,
                "mask_overlap_score": np.nan,
                "motion_score": np.nan,
                "projection_support_score": np.nan,
                "is_stationary_like_before_assignment": True,
                "is_anchor_particle": True,
            })

        return accepted_detection_ids, accepted_particle_ids
    
    def _assign_frame(self, frame_idx, detections):
        accepted_detection_ids = set()
        accepted_particle_ids = set()
        
        anchor_detection_ids, anchor_particle_ids = self._assign_anchors_first(
            frame_idx=frame_idx,
            detections=detections
            )
        
        accepted_detection_ids.update(anchor_detection_ids)
        accepted_particle_ids.update(anchor_particle_ids)
        
        remaining_detections = [
            det for det in detections
            if int(det.local_detection_index) not in accepted_detection_ids
            ]

        accepted = self._select_assignments(remaining_detections)

        for item in accepted:
            det = item["detection"]
            pid = int(item["particle_id"])
            particle = self.particles[pid]

            if particle.state == "lost":
                state_note = "reclaimed_anchor" if particle.is_anchor else "reclaimed"
                self.event_rows.append({
                    "frame": int(frame_idx),
                    "particle_id": pid,
                    "event_type": state_note,
                    "details": f"reclaimed after {particle.missed_frames} missed frame(s)"
                    })
            else:
                state_note = "continued"

            particle.add_detection(det, state_note=state_note)

            accepted_detection_ids.add(int(det.local_detection_index))
            accepted_particle_ids.add(pid)

            self.assignment_rows.append({
                "frame": int(det.frame),
                "label_id": int(det.label_id),
                "track_id": pid,
                "particle_id": pid,
                "y": float(det.y),
                "x": float(det.x),
                "area": float(det.area),
                "bbox_y0": int(det.bbox_y0),
                "bbox_x0": int(det.bbox_x0),
                "bbox_y1": int(det.bbox_y1),
                "bbox_x1": int(det.bbox_x1),
                "mean_intensity": float(det.mean_intensity),
                "max_intensity": float(det.max_intensity),
                "eccentricity": float(det.eccentricity),
                "assignment_state": state_note,
                "assignment_score": float(item["score_info"]["final_score"]),
                "distance_score": float(item["score_info"]["distance_score"]),
                "area_score": float(item["score_info"]["area_score"]),
                "mask_overlap_score": float(item["score_info"]["mask_overlap_score"]),
                "motion_score": float(item["score_info"]["motion_score"]),
                "projection_support_score": float(item["score_info"]["projection_support_score"]),
                "is_stationary_like_before_assignment": bool(item["score_info"]["is_stationary_like"]),
                "is_anchor_particle": bool(item["score_info"]["is_anchor"]),
                })

        for particle in self.particles.values():
            if particle.state not in ("active", "lost"):
                continue

            if int(particle.particle_id) in accepted_particle_ids:
                continue

            last_det = particle.last_detection()
            if last_det is None:
                continue

            if int(last_det.frame) >= int(frame_idx):
                continue

            particle.mark_missed()

            max_missed_allowed = self.max_missed_frames
            if particle.is_anchor:
                max_missed_allowed += self.anchor_extra_missed_frames

            if particle.missed_frames > max_missed_allowed:
                particle.mark_ended(frame_idx - 1)
                self.event_rows.append({
                    "frame": int(frame_idx),
                    "particle_id": int(particle.particle_id),
                    "event_type": "ended_anchor" if particle.is_anchor else "ended",
                    "details": f"ended after {particle.missed_frames} missed frame(s)"
                    })
            else:
                self.event_rows.append({
                    "frame": int(frame_idx),
                    "particle_id": int(particle.particle_id),
                    "event_type": "lost_anchor" if particle.is_anchor else "lost",
                    "details": f"missed_frames={particle.missed_frames}"
                    })

        for det in detections:
            if int(det.local_detection_index) in accepted_detection_ids:
                continue

            anchor_info = self._anchor_info_for_detection(det)

            if anchor_info is not None:
                particle, reused_existing_anchor = self._create_or_reuse_anchor_particle(
                    detection=det,
                    anchor_info=anchor_info
                )

                state = "reused_anchor_key" if reused_existing_anchor else "new_anchor"
                is_anchor_flag = True

                if reused_existing_anchor:
                    self.event_rows.append({
                        "frame": int(frame_idx),
                        "particle_id": int(particle.particle_id),
                        "event_type": "anchor_key_reused",
                        "details": f"anchor key {particle.anchor_key} reused"
                    })

            else:
                particle = self._create_particle(det)
                state = "new_particle"
                is_anchor_flag = False

            accepted_detection_ids.add(int(det.local_detection_index))
            accepted_particle_ids.add(int(particle.particle_id))

            self.assignment_rows.append({
                "frame": int(det.frame),
                "label_id": int(det.label_id),
                "track_id": int(particle.particle_id),
                "particle_id": int(particle.particle_id),
                "y": float(det.y),
                "x": float(det.x),
                "area": float(det.area),
                "bbox_y0": int(det.bbox_y0),
                "bbox_x0": int(det.bbox_x0),
                "bbox_y1": int(det.bbox_y1),
                "bbox_x1": int(det.bbox_x1),
                "mean_intensity": float(det.mean_intensity),
                "max_intensity": float(det.max_intensity),
                "eccentricity": float(det.eccentricity),
                "assignment_state": state,
                "assignment_score": 1.0,
                "distance_score": np.nan,
                "area_score": np.nan,
                "mask_overlap_score": np.nan,
                "motion_score": np.nan,
                "projection_support_score": np.nan,
                "is_stationary_like_before_assignment": is_anchor_flag,
                "is_anchor_particle": is_anchor_flag,
                })

    def track(self, detection_maps, progress_window=None, progress_start=25, progress_end=55):
        n_frames = len(detection_maps)

        for frame_idx in range(n_frames):
            frame_detections = []

            for detection_index, d in enumerate(detection_maps[frame_idx]):
                frame_detections.append(
                    DetectionRecord(
                        frame=int(d["frame"]),
                        label_id=int(d["label_id"]),
                        local_detection_index=int(detection_index),
                        y=float(d["y"]),
                        x=float(d["x"]),
                        area=float(d["area"]),
                        bbox_y0=int(d["bbox_y0"]),
                        bbox_x0=int(d["bbox_x0"]),
                        bbox_y1=int(d["bbox_y1"]),
                        bbox_x1=int(d["bbox_x1"]),
                        mean_intensity=float(d["mean_intensity"]),
                        max_intensity=float(d["max_intensity"]),
                        eccentricity=float(d["eccentricity"]),
                        coords=np.asarray(d["coords"], dtype=int),
                        )
                    )
                
        
            self._assign_frame(frame_idx, frame_detections)

            if progress_window is not None:
                progress_value = progress_start + ((frame_idx + 1) / n_frames) * (progress_end - progress_start)
                progress_window.set_progress(
                    progress_value,
                    text=f"Assigning persistent identities... frame {frame_idx + 1}/{n_frames}"
                    )

   
        final_frame = n_frames - 1

        for particle in self.particles.values():
            if particle.state in ("active", "lost"):
                particle.mark_ended(final_frame)

        detections_df = pd.DataFrame(self.assignment_rows)
        events_df = pd.DataFrame(self.event_rows)

        if len(detections_df) > 0:
            detections_df = detections_df.sort_values(
                ["track_id", "frame", "label_id"]
                ).reset_index(drop=True)

        if len(events_df) == 0:
            events_df = pd.DataFrame(
                columns=["frame", "particle_id", "event_type", "details"]
                )
        else:
                events_df = events_df.sort_values(
                    ["frame", "particle_id", "event_type"]
                    ).reset_index(drop=True)

        return detections_df, events_df


# =============================================================================
# FILTERS / METRICS / QUALITY
# =============================================================================

class TrackFilter:
    @staticmethod
    def filter_short_tracks(detections_df, min_length):
        if detections_df is None or len(detections_df) == 0:
            return detections_df.copy()

        counts = detections_df.groupby("track_id").size()
        valid_track_ids = counts[counts >= int(min_length)].index.tolist()

        if len(valid_track_ids) == 0:
            return detections_df.iloc[0:0].copy()

        return detections_df[detections_df["track_id"].isin(valid_track_ids)].copy()


class MSDCalculator:
    @staticmethod
    def compute_msd_curves(detections_df, max_lag_frames=10):
        if detections_df is None or len(detections_df) == 0 or "track_id" not in detections_df.columns:
            return pd.DataFrame(columns=[
                "track_id", "lag_frames", "lag_time_s",
                "msd_pixels2", "msd_um2", "n_pairs"
            ])

        rows = []

        for track_id, group in detections_df.groupby("track_id"):
            group = group.sort_values("frame").copy()
            points = group[["y", "x"]].to_numpy(dtype=float)
            frames = group["frame"].to_numpy(dtype=int)

            n = len(points)
            if n < 2:
                continue

            max_possible_lag = min(int(max_lag_frames), n - 1)

            for lag in range(1, max_possible_lag + 1):
                squared_displacements = []

                for i in range(0, n - lag):
                    frame_delta = int(frames[i + lag] - frames[i])
                    if frame_delta != lag:
                        continue

                    dy = float(points[i + lag, 0] - points[i, 0])
                    dx = float(points[i + lag, 1] - points[i, 1])
                    squared_displacements.append(dx * dx + dy * dy)

                if len(squared_displacements) == 0:
                    continue

                msd_pixels2 = float(np.mean(squared_displacements))
                msd_um2 = msd_pixels2 * (PIXEL_SIZE_UM ** 2)

                rows.append({
                    "track_id": int(track_id),
                    "lag_frames": int(lag),
                    "lag_time_s": float(lag * FRAME_INTERVAL_S),
                    "msd_pixels2": msd_pixels2,
                    "msd_um2": msd_um2,
                    "n_pairs": int(len(squared_displacements))
                })

        return pd.DataFrame(rows)

    @staticmethod
    def add_msd_summary_columns(summary_df, msd_df):
        if len(summary_df) == 0:
            return summary_df.copy()

        out = summary_df.copy()

        for lag in [1, 2, 3]:
            out[f"msd_lag{lag}_um2"] = np.nan

        out["msd_initial_slope_um2_per_s"] = np.nan

        if len(msd_df) == 0:
            return out

        for track_id, group in msd_df.groupby("track_id"):
            group = group.sort_values("lag_frames")
            idx = out.index[out["track_id"] == int(track_id)]
            if len(idx) == 0:
                continue
            idx = idx[0]

            for lag in [1, 2, 3]:
                lag_rows = group[group["lag_frames"] == lag]
                if len(lag_rows) > 0:
                    out.at[idx, f"msd_lag{lag}_um2"] = float(lag_rows.iloc[0]["msd_um2"])

            first_points = group.head(3)
            if len(first_points) >= 2:
                x = first_points["lag_time_s"].to_numpy(dtype=float)
                y = first_points["msd_um2"].to_numpy(dtype=float)
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    out.at[idx, "msd_initial_slope_um2_per_s"] = float(slope)
                except Exception:
                    pass

        return out

    @staticmethod
    def compute_global_msd_curve(msd_df):
        if len(msd_df) == 0:
            return pd.DataFrame(columns=[
                "lag_frames",
                "lag_time_s",
                "mean_msd_um2",
                "std_msd_um2",
                "sem_msd_um2",
                "n_tracks"
            ])

        rows = []

        for lag_frames, group in msd_df.groupby("lag_frames"):
            values = group["msd_um2"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            n_tracks = int(group["track_id"].nunique())
            mean_val = float(np.mean(values))

            if len(values) > 1:
                std_val = float(np.std(values, ddof=1))
            else:
                std_val = 0.0

            sem_val = float(std_val / np.sqrt(len(values))) if len(values) > 0 else 0.0

            rows.append({
                "lag_frames": int(lag_frames),
                "lag_time_s": float(group["lag_time_s"].iloc[0]),
                "mean_msd_um2": mean_val,
                "std_msd_um2": std_val,
                "sem_msd_um2": sem_val,
                "n_tracks": n_tracks
            })

        return pd.DataFrame(rows).sort_values("lag_frames").reset_index(drop=True)


class TrackMetricsCalculator:
    @staticmethod
    def compute_metrics(detections_df):
        if detections_df is None or len(detections_df) == 0 or "track_id" not in detections_df.columns:
            return pd.DataFrame(), pd.DataFrame()

        summary_rows = []
        step_rows = []

        for track_id, group in detections_df.groupby("track_id"):
            group = group.sort_values("frame").copy()
            points = group[["y", "x"]].to_numpy()
            frames = group["frame"].to_numpy()

            first_row = group.iloc[0]
            last_row = group.iloc[-1]

            dx_net = float(last_row["x"] - first_row["x"])
            dy_net = float(last_row["y"] - first_row["y"])
            net_displacement_um = float(np.sqrt(dx_net ** 2 + dy_net ** 2)) * PIXEL_SIZE_UM

            step_distances_um = []
            step_speeds_um_per_s = []
            step_apparent_D_um2_per_s = []

            if len(points) > 1:
                for i in range(len(points) - 1):
                    y0, x0 = points[i]
                    y1, x1 = points[i + 1]

                    frame_from = int(frames[i])
                    frame_to = int(frames[i + 1])
                    gap_frames = int(frame_to - frame_from - 1)
                    delta_time_s = float((frame_to - frame_from) * FRAME_INTERVAL_S)

                    step_distance_pixels = float(np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2))
                    step_distance_um = step_distance_pixels * PIXEL_SIZE_UM

                    if delta_time_s > 0:
                        speed_um_per_s = step_distance_um / delta_time_s
                        apparent_D_um2_per_s = (step_distance_um ** 2) / (4.0 * delta_time_s)
                    else:
                        speed_um_per_s = 0.0
                        apparent_D_um2_per_s = 0.0

                    step_distances_um.append(step_distance_um)
                    step_speeds_um_per_s.append(speed_um_per_s)
                    step_apparent_D_um2_per_s.append(apparent_D_um2_per_s)

                    step_rows.append({
                        "track_id": int(track_id),
                        "frame_from": frame_from,
                        "frame_to": frame_to,
                        "gap_frames": gap_frames,
                        "delta_time_s": delta_time_s,
                        "step_distance_um": step_distance_um,
                        "speed_um_per_s": speed_um_per_s,
                        "apparent_D_um2_per_s": apparent_D_um2_per_s,
                    })

            total_path_length_um = float(np.sum(step_distances_um)) if len(step_distances_um) > 0 else 0.0
            is_anchor_particle = bool(group["is_anchor_particle"].iloc[0]) if "is_anchor_particle" in group.columns else False
            is_moving_particle = bool(total_path_length_um > 0.5 and not is_anchor_particle)

            summary_rows.append({
                "track_id": int(track_id),
                "n_points": int(len(group)),
                "first_frame_detected": int(group["frame"].min()),
                "last_frame_detected": int(group["frame"].max()),
                "net_displacement_um": net_displacement_um,
                "total_path_length_um": total_path_length_um,
                "mean_step_distance_um": float(np.mean(step_distances_um)) if len(step_distances_um) > 0 else 0.0,
                "median_step_distance_um": float(np.median(step_distances_um)) if len(step_distances_um) > 0 else 0.0,
                "max_step_distance_um": float(np.max(step_distances_um)) if len(step_distances_um) > 0 else 0.0,
                "mean_speed_um_per_s": float(np.mean(step_speeds_um_per_s)) if len(step_speeds_um_per_s) > 0 else 0.0,
                "median_speed_um_per_s": float(np.median(step_speeds_um_per_s)) if len(step_speeds_um_per_s) > 0 else 0.0,
                "max_speed_um_per_s": float(np.max(step_speeds_um_per_s)) if len(step_speeds_um_per_s) > 0 else 0.0,
                "mean_apparent_D_um2_per_s": float(np.mean(step_apparent_D_um2_per_s)) if len(step_apparent_D_um2_per_s) > 0 else 0.0,
                "median_apparent_D_um2_per_s": float(np.median(step_apparent_D_um2_per_s)) if len(step_apparent_D_um2_per_s) > 0 else 0.0,
                "max_apparent_D_um2_per_s": float(np.max(step_apparent_D_um2_per_s)) if len(step_apparent_D_um2_per_s) > 0 else 0.0,
                "n_reclaimed_points": int(group["assignment_state"].isin(["reclaimed", "reclaimed_anchor"]).sum()),
                "n_new_particle_points": int(group["assignment_state"].isin(["new_particle", "new_anchor"]).sum()),
                "is_anchor_particle": bool(is_anchor_particle),
                "is_moving_particle": bool(is_moving_particle),
            })

        return pd.DataFrame(summary_rows), pd.DataFrame(step_rows)


class QualityChecker:
    @staticmethod
    def build_quality_table(
        detections_all,
        detections_filtered,
        summary_filtered,
        events_df,
        min_track_length,
        sum_regions_df=None,
        anchor_df=None
    ):
        def safe_median(series):
            series = np.asarray(series, dtype=float)
            series = series[np.isfinite(series)]
            return float(np.median(series)) if len(series) > 0 else 0.0

        n_sum_regions = int(len(sum_regions_df)) if sum_regions_df is not None and len(sum_regions_df) > 0 else 0
        n_good_anchors = int((anchor_df["is_anchor"] == True).sum()) if anchor_df is not None and len(anchor_df) > 0 else 0

        rows = [
            {"metric": "analysis_min_track_length", "value": int(min_track_length)},
            {"metric": "rows_before_filter", "value": int(len(detections_all))},
            {"metric": "rows_after_filter", "value": int(len(detections_filtered))},
            {"metric": "tracks_after_filter", "value": int(summary_filtered["track_id"].nunique()) if len(summary_filtered) > 0 else 0},
            {"metric": "sum_regions_detected", "value": int(n_sum_regions)},
            {"metric": "validated_anchors", "value": int(n_good_anchors)},
            {"metric": "created_events", "value": int(events_df["event_type"].isin(["created", "created_anchor"]).sum()) if len(events_df) > 0 else 0},
            {"metric": "reclaimed_events", "value": int(events_df["event_type"].isin(["reclaimed", "reclaimed_anchor"]).sum()) if len(events_df) > 0 else 0},
            {"metric": "lost_events", "value": int(events_df["event_type"].isin(["lost", "lost_anchor"]).sum()) if len(events_df) > 0 else 0},
            {"metric": "ended_events", "value": int(events_df["event_type"].isin(["ended", "ended_anchor"]).sum()) if len(events_df) > 0 else 0},
            {"metric": "median_track_duration", "value": safe_median(summary_filtered["n_points"]) if len(summary_filtered) > 0 else 0.0},
            {"metric": "median_total_path_length_um", "value": safe_median(summary_filtered["total_path_length_um"]) if len(summary_filtered) > 0 else 0.0},
            {"metric": "median_mean_apparent_D_um2_per_s", "value": safe_median(summary_filtered["mean_apparent_D_um2_per_s"]) if len(summary_filtered) > 0 else 0.0},
        ]

        return pd.DataFrame(rows)


# =============================================================================
# RESULT SAVER
# =============================================================================

class ResultSaver:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def save_detection_table(self, detections_df, suffix):
        detections_df.to_csv(
            os.path.join(self.output_folder, f"detections_with_tracks_{suffix}.csv"),
            index=False
        )

    def save_events_table(self, events_df):
        events_df.to_csv(
            os.path.join(self.output_folder, "tracking_events.csv"),
            index=False
        )

    def save_sum_regions_table(self, regions_df):
        regions_df.to_csv(
            os.path.join(self.output_folder, "sum_projection_regions.csv"),
            index=False
        )

    def save_anchor_table(self, anchor_df):
        anchor_df.to_csv(
            os.path.join(self.output_folder, "validated_anchor_candidates.csv"),
            index=False
        )

    def save_calibration_table(self, calibration_df):
        calibration_df.to_csv(
            os.path.join(self.output_folder, "anchor_based_parameter_calibration.csv"),
            index=False
        )

    def save_settings_table(self, settings_before, settings_after):
        rows = []
        all_keys = sorted(set(settings_before.keys()).union(set(settings_after.keys())))
        for key in all_keys:
            rows.append({
                "parameter": key,
                "value_before_calibration": settings_before.get(key, np.nan),
                "value_after_calibration": settings_after.get(key, np.nan),
            })

        pd.DataFrame(rows).to_csv(
            os.path.join(self.output_folder, "settings_used.csv"),
            index=False
        )

    def save_quality_checks(self, quality_df):
        quality_df.to_csv(
            os.path.join(self.output_folder, "quality_checks.csv"),
            index=False
        )

    def save_metrics_tables(self, summary_df, steps_df, suffix):
        summary_df.to_csv(
            os.path.join(self.output_folder, f"track_summary_{suffix}.csv"),
            index=False
        )
        steps_df.to_csv(
            os.path.join(self.output_folder, f"track_step_distances_{suffix}.csv"),
            index=False
        )

    def save_msd_tables(self, msd_df, global_msd_df, suffix):
        msd_df.to_csv(
            os.path.join(self.output_folder, f"track_msd_curves_{suffix}.csv"),
            index=False
        )
        global_msd_df.to_csv(
            os.path.join(self.output_folder, f"global_msd_curve_{suffix}.csv"),
            index=False
        )

    def save_excel_workbook(
        self,
        detections_all,
        detections_filtered,
        summary_filtered,
        steps_filtered,
        msd_filtered,
        global_msd_filtered,
        events_df,
        quality_df,
        sum_regions_df,
        anchor_df,
        calibration_df,
        settings_before,
        settings_after
    ):
        excel_path = os.path.join(self.output_folder, "tracking_results.xlsx")

        engine_to_use = None
        try:
            import openpyxl  # noqa: F401
            engine_to_use = "openpyxl"
        except Exception:
            try:
                import xlsxwriter  # noqa: F401
                engine_to_use = "xlsxwriter"
            except Exception:
                engine_to_use = None

        if engine_to_use is None:
            return False

        settings_rows = []
        all_keys = sorted(set(settings_before.keys()).union(set(settings_after.keys())))
        for key in all_keys:
            settings_rows.append({
                "parameter": key,
                "value_before_calibration": settings_before.get(key, np.nan),
                "value_after_calibration": settings_after.get(key, np.nan),
            })
        settings_df = pd.DataFrame(settings_rows)

        with pd.ExcelWriter(excel_path, engine=engine_to_use) as writer:
            detections_all.to_excel(writer, sheet_name="all_detections", index=False)
            detections_filtered.to_excel(writer, sheet_name="filtered_detections", index=False)
            summary_filtered.to_excel(writer, sheet_name="track_summary", index=False)
            steps_filtered.to_excel(writer, sheet_name="track_steps", index=False)
            msd_filtered.to_excel(writer, sheet_name="track_msd", index=False)
            global_msd_filtered.to_excel(writer, sheet_name="global_msd", index=False)
            events_df.to_excel(writer, sheet_name="events", index=False)
            quality_df.to_excel(writer, sheet_name="quality_checks", index=False)
            if sum_regions_df is not None and len(sum_regions_df) > 0:
                sum_regions_df.to_excel(writer, sheet_name="sum_regions", index=False)
            if anchor_df is not None and len(anchor_df) > 0:
                anchor_df.to_excel(writer, sheet_name="anchors", index=False)
            if calibration_df is not None and len(calibration_df) > 0:
                calibration_df.to_excel(writer, sheet_name="calibration", index=False)
            settings_df.to_excel(writer, sheet_name="settings_used", index=False)

        return True

    def save_summary_plot(self, reference_stack, detections_df, summary_df, suffix, min_track_length_to_draw):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(reference_stack[0], cmap="gray")

        if detections_df is not None and len(detections_df) > 0 and len(summary_df) > 0:
            valid_track_ids = set(summary_df.loc[summary_df["n_points"] >= min_track_length_to_draw, "track_id"].tolist())

            for track_id, group in detections_df.groupby("track_id"):
                if track_id not in valid_track_ids:
                    continue

                color = np.array(TrackColorManager.get_color(track_id), dtype=float) / 255.0
                group = group.sort_values("frame")
                ax.plot(group["x"], group["y"], linewidth=0.8, color=color)

        ax.set_title(f"Track summary on first frame ({suffix})")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_folder, f"track_summary_plot_{suffix}.png"), dpi=300)
        plt.close(fig)

    def save_histogram(self, values, title, xlabel, filename, bins=30):
        fig, ax = plt.subplots(figsize=(7, 5))
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_folder, filename), dpi=300)
        plt.close(fig)

    def save_track_histograms(self, summary_df, steps_df, suffix):
        self.save_histogram(
            summary_df["total_path_length_um"].to_numpy() if len(summary_df) > 0 else [],
            f"Distribution of total path lengths ({suffix})",
            "Total path length (µm)",
            f"hist_total_path_lengths_um_{suffix}.png"
        )
        self.save_histogram(
            steps_df["step_distance_um"].to_numpy() if len(steps_df) > 0 else [],
            f"Distribution of frame-to-frame step distances ({suffix})",
            "Step distance (µm)",
            f"hist_step_distances_um_{suffix}.png"
        )
        self.save_histogram(
            summary_df["mean_speed_um_per_s"].to_numpy() if len(summary_df) > 0 else [],
            f"Distribution of mean track speeds ({suffix})",
            "Mean speed (µm/s)",
            f"hist_mean_speed_um_per_s_{suffix}.png"
        )
        self.save_histogram(
            summary_df["mean_apparent_D_um2_per_s"].to_numpy() if len(summary_df) > 0 else [],
            f"Distribution of mean apparent diffusion coefficient ({suffix})",
            "Mean apparent D (µm²/s)",
            f"hist_mean_apparent_D_um2_per_s_{suffix}.png"
        )
        self.save_histogram(
            steps_df["apparent_D_um2_per_s"].to_numpy() if len(steps_df) > 0 else [],
            f"Distribution of step apparent diffusion coefficient ({suffix})",
            "Step apparent D (µm²/s)",
            f"hist_step_apparent_D_um2_per_s_{suffix}.png"
        )
        self.save_histogram(
            summary_df["n_points"].to_numpy() if len(summary_df) > 0 else [],
            f"Distribution of track duration ({suffix})",
            "Track duration (detections)",
            f"hist_track_duration_{suffix}.png"
        )

    def save_msd_histograms(self, summary_df, suffix):
        for lag in [1, 2, 3]:
            col = f"msd_lag{lag}_um2"
            if col in summary_df.columns:
                self.save_histogram(
                    summary_df[col].to_numpy() if len(summary_df) > 0 else [],
                    f"Distribution of MSD lag {lag} ({suffix})",
                    f"MSD lag {lag} (µm²)",
                    f"hist_msd_lag{lag}_um2_{suffix}.png"
                )

    def save_global_msd_plot(self, global_msd_df, suffix):
        fig, ax = plt.subplots(figsize=(7, 5))

        if len(global_msd_df) > 0:
            x = global_msd_df["lag_time_s"].to_numpy(dtype=float)
            y = global_msd_df["mean_msd_um2"].to_numpy(dtype=float)
            yerr = global_msd_df["std_msd_um2"].to_numpy(dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linestyle="-",
                capsize=4
            )

        ax.set_title(f"Global MSD curve ({suffix})")
        ax.set_xlabel("Lag time (s)")
        ax.set_ylabel("Mean MSD (µm²)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_folder, f"global_msd_curve_{suffix}.png"), dpi=300)
        plt.close(fig)

    def save_projection_stack(self, projection_stack, suffix):
        output_path = os.path.join(self.output_folder, f"{suffix}.tif")
        tiff.imwrite(output_path, np.asarray(projection_stack))
        return output_path

    def save_rgb_movie(self, rgb_stack, suffix):
        output_path = os.path.join(self.output_folder, f"{suffix}.tif")
        with tiff.TiffWriter(output_path, bigtiff=True) as writer:
            if rgb_stack.ndim == 3:
                writer.write(rgb_stack.astype(np.uint8), photometric="rgb")
            else:
                for i in range(rgb_stack.shape[0]):
                    writer.write(rgb_stack[i].astype(np.uint8), photometric="rgb")
        return output_path

    def save_overlay_movie(
        self,
        output_path,
        overlay_stack,
        label_maps,
        detections_df,
        renderer,
        min_track_length_to_draw=1,
        track_ids_to_keep=None
    ):
        n_frames, height, width = overlay_stack.shape
        line_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        previous_point_by_track = {}

        if detections_df is None or len(detections_df) == 0:
            detections_df = pd.DataFrame(columns=["frame", "track_id", "y", "x", "label_id"])

        detections_to_draw = detections_df.copy()

        if track_ids_to_keep is not None:
            keep = set(int(t) for t in track_ids_to_keep)
            detections_to_draw = detections_to_draw[detections_to_draw["track_id"].isin(keep)].copy()

        frame_groups = {
            int(frame): group.copy()
            for frame, group in detections_to_draw.groupby("frame")
        } if len(detections_to_draw) > 0 else {}

        track_lengths = (
            detections_to_draw.groupby("track_id").size().to_dict()
            if len(detections_to_draw) > 0 else {}
        )

        with tiff.TiffWriter(output_path, bigtiff=True) as writer:
            empty_frame_df = detections_to_draw.iloc[0:0].copy()

            for frame_index in range(n_frames):
                overlay_frame = overlay_stack[frame_index]
                label_map_frame = label_maps[frame_index]
                frame_detections = frame_groups.get(frame_index, empty_frame_df)

                if len(frame_detections) > 0:
                    renderer.draw_new_segments_only(
                        line_canvas=line_canvas,
                        frame_detections=frame_detections,
                        previous_point_by_track=previous_point_by_track,
                        track_lengths=track_lengths,
                        min_track_length_to_draw=min_track_length_to_draw
                    )

                overlay = renderer.render_outline_and_lines_frame(
                    raw_frame=overlay_frame,
                    label_map_frame=label_map_frame,
                    frame_detections=frame_detections,
                    line_canvas=line_canvas
                ).astype(np.uint8)

                writer.write(overlay, photometric="rgb")


# =============================================================================
# PIPELINE
# =============================================================================

class TrackingPipeline:
    def __init__(self, settings):
        self.settings_before_calibration = settings.copy()
        self.settings_used = settings.copy()

        self.detector = DetectionExtractor(
            min_area=settings["min_area"],
            max_area=settings["max_area"],
            use_touching_split=settings["use_touching_split"],
            touch_split_min_peak_distance=settings["touch_split_min_peak_distance"],
        )
        self.renderer = OverlayRenderer(show_track_lines=True)

    @staticmethod
    def _subtract_mask_from_stack(stack, mask2d):
        cleaned = np.asarray(stack).copy()
        cleaned[:, mask2d] = 0
        return cleaned

    @staticmethod
    def _get_top_n_longest_distance_track_ids(summary_df, n=3):
        if summary_df is None or len(summary_df) == 0:
            return []
        return (
            summary_df.sort_values("total_path_length_um", ascending=False)
            .head(int(n))["track_id"]
            .astype(int)
            .tolist()
        )

    @staticmethod
    def _get_persistent_mixed_track_ids(summary_df, anchor_df, detections_filtered, top_n_moving=3, top_n_immobile=3):
        if summary_df is None or len(summary_df) == 0:
            return []

        moving = summary_df[summary_df["is_moving_particle"] == True].copy()
        moving_ids = []
        if len(moving) > 0:
            moving_ids = (
                moving.sort_values(["n_points", "total_path_length_um"], ascending=[False, False])
                .head(int(top_n_moving))["track_id"]
                .astype(int)
                .tolist()
            )

        immobile_ids = []
        immobile_summary = summary_df[summary_df["is_anchor_particle"] == True].copy()
        if len(immobile_summary) > 0:
            immobile_ids = (
                immobile_summary.sort_values(["n_points", "total_path_length_um"], ascending=[False, False])
                .head(int(top_n_immobile))["track_id"]
                .astype(int)
                .tolist()
            )

        out = []
        for tid in moving_ids + immobile_ids:
            if tid not in out:
                out.append(tid)

        return out

    def run(self, tracking_path, overlay_path, output_folder):
        progress = ProgressWindow(title="Identity-first particle tracking", maximum=100)

        try:
            TrackColorManager.reset()

            progress.set_progress(1, text="Loading stacks...")
            tracking_stack = StackLoader.load_tiff_stack(tracking_path)
            overlay_stack = StackLoader.load_tiff_stack(overlay_path)
            StackLoader.validate_same_shape(tracking_stack, overlay_stack)

            saver = ResultSaver(output_folder)

            progress.set_progress(8, text="Detecting particles on original frames...")
            detections_raw_df, label_maps, detection_maps = self.detector.detect_in_stack(
                raw_stack=tracking_stack,
                mask_stack=tracking_stack,
                progress_window=progress,
                progress_start=8,
                progress_end=28
            )

            progress.set_progress(30, text="Building SUM immobile regions...")
            sum_region_builder = SumProjectionImmobileRegionBuilder(
                min_area=self.settings_before_calibration["sum_region_min_area"],
                max_area=self.settings_before_calibration["sum_region_max_area"],
                dilation_radius=self.settings_before_calibration["sum_region_dilation_radius"],
                threshold_percentile=self.settings_before_calibration["sum_region_threshold_percentile"],
                keep_top_fraction=self.settings_before_calibration["sum_core_keep_fraction"],
            )
            sum_projection, sum_region_label_map, sum_region_mask, sum_regions_df = sum_region_builder.build(tracking_stack)

            saver.save_sum_regions_table(sum_regions_df)

            if self.settings_before_calibration["save_sum_projection_region_movie"]:
                sum_rgb = SimpleImageRenderer.sum_region_rgb(sum_projection, sum_region_label_map)
                saver.save_rgb_movie(sum_rgb, suffix="sum_projection_immobile_regions")

            progress.set_progress(38, text="Subtracting SUM core immobile regions before rolling MAX support...")
            cleaned_support_stack = self._subtract_mask_from_stack(tracking_stack, sum_region_mask)

            if self.settings_before_calibration["save_cleaned_support_stack_movie"]:
                saver.save_projection_stack(cleaned_support_stack, suffix="support_stack_after_sum_region_subtraction")

            progress.set_progress(45, text="Building rolling MAX support from cleaned stack...")
            projection_builder = ProjectionSupportBuilder(
                splitter=self.detector.splitter,
                min_area=self.settings_before_calibration["min_area"],
                window_size=self.settings_before_calibration["rolling_projection_window_size"],
                support_threshold=self.settings_before_calibration["projection_support_threshold"],
                corridor_radius=self.settings_before_calibration["projection_corridor_radius"],
            )
            projection_builder.build(cleaned_support_stack)

            if self.settings_before_calibration["save_rolling_projection_movie"]:
                saver.save_projection_stack(
                    projection_builder.max_projection_stack,
                    suffix=f"rolling_max_projection_movie_cleaned_w{self.settings_before_calibration['rolling_projection_window_size']}"
                )

            if self.settings_before_calibration["save_support_movie"]:
                support_movie = SimpleImageRenderer.support_movie_rgb(
                    projection_builder.max_projection_stack,
                    projection_builder.support_masks
                )
                saver.save_rgb_movie(
                    support_movie,
                    suffix=f"projection_support_movie_cleaned_w{self.settings_before_calibration['rolling_projection_window_size']}"
                )

            progress.set_progress(55, text="Validating true anchors from original detections...")
            if self.settings_before_calibration["use_anchor_validation"]:
                anchor_validator = AnchorValidator(
                    detection_radius=self.settings_before_calibration["anchor_detection_radius"],
                    min_presence_fraction=self.settings_before_calibration["anchor_min_presence_fraction"],
                    max_jitter_pixels=self.settings_before_calibration["anchor_max_jitter_pixels"],
                    min_frames_present=self.settings_before_calibration["anchor_min_frames_present"],
                    max_allowed_gap=self.settings_before_calibration["anchor_max_allowed_gap"],
                )
                anchor_df = anchor_validator.validate(
                    regions_df=sum_regions_df,
                    detection_maps=detection_maps,
                    n_frames=tracking_stack.shape[0]
                )
            else:
                anchor_df = pd.DataFrame()

            saver.save_anchor_table(anchor_df)

            if self.settings_before_calibration["use_anchor_calibration"] and len(anchor_df) > 0:
                calibrator = AutoParameterCalibrator(
                    calibration_strength=self.settings_before_calibration["anchor_calibration_strength"]
                )
                self.settings_used, calibration_df = calibrator.calibrate(
                    settings=self.settings_before_calibration,
                    anchor_df=anchor_df
                )
            else:
                self.settings_used = self.settings_before_calibration.copy()
                calibration_df = pd.DataFrame()

            saver.save_calibration_table(calibration_df)
            saver.save_settings_table(
                settings_before=self.settings_before_calibration,
                settings_after=self.settings_used
            )

            progress.set_progress(62, text="Tracking persistent identities...")
            tracker = IdentityFirstTracker(
                settings=self.settings_used,
                projection_builder=projection_builder,
                anchor_df=anchor_df
            )
            detections_all, events_df = tracker.track(
                detection_maps=detection_maps,
                progress_window=progress,
                progress_start=62,
                progress_end=82
            )

            if len(detections_all) > 0:
                all_track_ids = detections_all["track_id"].unique()
                TrackColorManager.build_color_map_from_tracks(all_track_ids)

            min_track_length = int(self.settings_used["min_track_length"])
            detections_filtered = TrackFilter.filter_short_tracks(detections_all, min_track_length)

            progress.set_progress(84, text="Computing metrics...")
            summary_filtered, steps_filtered = TrackMetricsCalculator.compute_metrics(detections_filtered)
            msd_filtered = MSDCalculator.compute_msd_curves(
                detections_filtered,
                max_lag_frames=self.settings_used["max_msd_lag"]
            )
            summary_filtered = MSDCalculator.add_msd_summary_columns(summary_filtered, msd_filtered)
            global_msd_filtered = MSDCalculator.compute_global_msd_curve(msd_filtered)

            saver.save_detection_table(detections_all, "all")
            saver.save_detection_table(detections_filtered, "filtered")
            saver.save_events_table(events_df)
            saver.save_metrics_tables(summary_filtered, steps_filtered, "filtered")
            saver.save_msd_tables(msd_filtered, global_msd_filtered, "filtered")
            saver.save_track_histograms(summary_filtered, steps_filtered, "filtered")
            saver.save_msd_histograms(summary_filtered, "filtered")
            saver.save_global_msd_plot(global_msd_filtered, "filtered")
            saver.save_summary_plot(
                tracking_stack,
                detections_filtered,
                summary_filtered,
                "filtered",
                min_track_length
            )

            quality_df = QualityChecker.build_quality_table(
                detections_all=detections_all,
                detections_filtered=detections_filtered,
                summary_filtered=summary_filtered,
                events_df=events_df,
                min_track_length=min_track_length,
                sum_regions_df=sum_regions_df,
                anchor_df=anchor_df
            )
            saver.save_quality_checks(quality_df)

            excel_saved = saver.save_excel_workbook(
                detections_all=detections_all,
                detections_filtered=detections_filtered,
                summary_filtered=summary_filtered,
                steps_filtered=steps_filtered,
                msd_filtered=msd_filtered,
                global_msd_filtered=global_msd_filtered,
                events_df=events_df,
                quality_df=quality_df,
                sum_regions_df=sum_regions_df,
                anchor_df=anchor_df,
                calibration_df=calibration_df,
                settings_before=self.settings_before_calibration,
                settings_after=self.settings_used
            )

            progress.set_progress(90, text="Writing full overlay movie...")
            full_overlay_output_path = os.path.join(output_folder, "tracked_identity_full_overlay.tif")
            saver.save_overlay_movie(
                output_path=full_overlay_output_path,
                overlay_stack=overlay_stack,
                label_maps=label_maps,
                detections_df=detections_all,
                renderer=self.renderer,
                min_track_length_to_draw=min_track_length,
                track_ids_to_keep=None
            )

            progress.set_progress(94, text="Writing 3 longest-distance particles movie...")
            longest_distance_ids = self._get_top_n_longest_distance_track_ids(summary_filtered, n=3)
            longest_distance_output_path = os.path.join(output_folder, "tracked_top3_longest_distance_overlay.tif")
            saver.save_overlay_movie(
                output_path=longest_distance_output_path,
                overlay_stack=overlay_stack,
                label_maps=label_maps,
                detections_df=detections_all,
                renderer=self.renderer,
                min_track_length_to_draw=1,
                track_ids_to_keep=longest_distance_ids
            )

            progress.set_progress(97, text="Writing persistent moving + immobile movie...")
            persistent_ids = self._get_persistent_mixed_track_ids(
                summary_df=summary_filtered,
                anchor_df=anchor_df,
                detections_filtered=detections_filtered,
                top_n_moving=int(self.settings_used["persistent_top_n_moving"]),
                top_n_immobile=int(self.settings_used["persistent_top_n_immobile"])
                )
            persistent_output_path = os.path.join(output_folder, "tracked_persistent_moving_and_immobile_overlay.tif")
            saver.save_overlay_movie(
                output_path=persistent_output_path,
                overlay_stack=overlay_stack,
                label_maps=label_maps,
                detections_df=detections_all,
                renderer=self.renderer,
                min_track_length_to_draw=1,
                track_ids_to_keep=persistent_ids
            )

            progress.set_progress(100, text="Finished.")
            progress.close()

            return {
                "detections_all": detections_all,
                "detections_filtered": detections_filtered,
                "summary_filtered": summary_filtered,
                "events_df": events_df,
                "sum_regions_df": sum_regions_df,
                "anchor_df": anchor_df,
                "settings_before_calibration": self.settings_before_calibration,
                "settings_used": self.settings_used,
                "excel_saved": excel_saved,
                "longest_distance_ids": longest_distance_ids,
                "persistent_ids": persistent_ids,
            }

        except Exception:
            progress.close()
            raise


# =============================================================================
# MAIN
# =============================================================================

def main():
    settings = SettingsDialog.ask()
    if settings is None:
        print("Settings dialog cancelled.")
        return

    tracking_path = FileSelector.select_tracking_file()
    if not tracking_path:
        print("No tracking stack selected.")
        return

    overlay_path = FileSelector.select_overlay_file()
    if not overlay_path:
        print("No overlay stack selected.")
        return

    output_folder = FileSelector.select_output_folder()
    if not output_folder:
        print("No output folder selected.")
        return

    pipeline = TrackingPipeline(settings=settings)

    try:
        results = pipeline.run(
            tracking_path=tracking_path,
            overlay_path=overlay_path,
            output_folder=output_folder
        )

        detections_all = results["detections_all"]
        detections_filtered = results["detections_filtered"]
        summary_filtered = results["summary_filtered"]
        events_df = results["events_df"]
        sum_regions_df = results["sum_regions_df"]
        anchor_df = results["anchor_df"]
        settings_before = results["settings_before_calibration"]
        settings_used = results["settings_used"]
        excel_saved = results["excel_saved"]
        longest_distance_ids = results.get("longest_distance_ids", [])
        persistent_ids = results.get("persistent_ids", [])

        if detections_all is None or len(detections_all) == 0:
            messagebox.showinfo("Finished", "Finished, but no particles were detected.")
        else:
            n_tracks_filtered = int(summary_filtered["track_id"].nunique()) if len(summary_filtered) > 0 else 0
            n_created = int(events_df["event_type"].isin(["created", "created_anchor"]).sum()) if len(events_df) > 0 else 0
            n_reclaimed = int(events_df["event_type"].isin(["reclaimed", "reclaimed_anchor"]).sum()) if len(events_df) > 0 else 0
            n_sum_regions = int(len(sum_regions_df)) if sum_regions_df is not None and len(sum_regions_df) > 0 else 0
            n_good_anchors = int((anchor_df["is_anchor"] == True).sum()) if anchor_df is not None and len(anchor_df) > 0 else 0

            changed_params = 0
            for key in settings_before.keys():
                if key in settings_used and settings_before[key] != settings_used[key]:
                    changed_params += 1

            messagebox.showinfo(
                "Finished",
                f"Finished.\n\n"
                f"Rows before length filter: {len(detections_all)}\n"
                f"Rows after length filter: {len(detections_filtered)}\n"
                f"Tracks after length filter: {n_tracks_filtered}\n\n"
                f"Created particles: {n_created}\n"
                f"Reclaimed identities: {n_reclaimed}\n"
                f"SUM regions detected: {n_sum_regions}\n"
                f"Validated anchors: {n_good_anchors}\n"
                f"Parameters changed by calibration: {changed_params}\n\n"
                f"3 longest-distance IDs: {longest_distance_ids}\n"
                f"Persistent mixed IDs: {persistent_ids}\n\n"
                f"Excel saved: {'yes' if excel_saved else 'no'}"
            )

    except Exception as error:
        messagebox.showerror("Error", str(error))
        raise


if __name__ == "__main__":
    main()
    