from __future__ import annotations

import math
import queue
import threading
from itertools import product
from pathlib import Path
from typing import Callable, TypeVar

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

    class _DummyBase:
        pass

    class _DummyTkModule:
        Toplevel = _DummyBase
        Tk = _DummyBase
        END = "end"

    class _DummyWidgetModule:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError("Tkinter is not available in this Python environment.")

    tk = _DummyTkModule()
    filedialog = _DummyWidgetModule()
    messagebox = _DummyWidgetModule()
    ttk = _DummyWidgetModule()

if TK_AVAILABLE:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        MPL_AVAILABLE = True
    except Exception:
        MPL_AVAILABLE = False
else:
    MPL_AVAILABLE = False

from .compute import (
    INCH_TO_M,
    InverseCandidate,
    LayerConfig,
    LoadedLayer,
    MaterialTable,
    UncertaintyConfig,
    build_uncertainty_scales,
    compute_angle_metrics,
    compute_angle_metrics_many,
    compute_stack_impedance_many,
    is_nominal_scale,
    make_sweep,
    normalize_backing,
    normalize_wave_polarization,
    validate_sweep_coverage,
)
from .io import (
    layer_config_from_dict,
    layer_config_to_dict,
    load_project_file,
    read_material_table,
    save_project_file,
    write_output,
)
from .plot import nearest_index, style_axis, style_colorbar

APP_ACRONYM = "FART"
APP_NAME = "Frequency-Angle Reflection Toolkit"
APP_TITLE = f"{APP_ACRONYM} - {APP_NAME}"
ABOUT_TEXT = (
    f"{APP_TITLE}\n\n"
    f"Acronym: {APP_NAME}\n\n"
    "Angle convention:\n"
    "0 deg = normal incidence (broadside)\n"
    "90 deg = grazing incidence\n\n"
    "Loss metric definitions:\n"
    "loss_db = -20*log10(|x|)\n"
    "metal_loss_db uses x = Gamma_metal\n"
    "air_loss_db uses x = Gamma_air\n"
    "insertion_loss_db uses x = S21\n\n"
    "Sign interpretation:\n"
    "positive loss_db => |x| < 1 (attenuation)\n"
    "zero loss_db => |x| = 1\n"
    "negative loss_db => |x| > 1 (effective gain/non-passive)"
)

LIGHT_THEME = {
    "window_bg": "#f5f6f8",
    "panel_bg": "#f5f6f8",
    "text": "#1f2933",
    "muted_text": "#4b5563",
    "field_bg": "#ffffff",
    "field_fg": "#111827",
    "field_disabled_bg": "#e5e7eb",
    "field_disabled_fg": "#6b7280",
    "button_bg": "#e5e7eb",
    "button_active_bg": "#d1d5db",
    "selection_bg": "#2563eb",
    "selection_fg": "#ffffff",
    "accent": "#2563eb",
    "preview_bg": "#f7f7f7",
    "preview_border": "#b0b0b0",
    "preview_outline": "#3a3a3a",
    "preview_text": "#404040",
    "preview_empty": "#5a5a5a",
    "preview_layer_text": "#1f1f1f",
    "preview_layer_border": "#ffffff",
    "layer_colors": [
        "#89c2ff",
        "#ffd166",
        "#90d39a",
        "#f4a6a6",
        "#c9b6ff",
        "#7fd8d8",
        "#ffb570",
        "#c2d36b",
    ],
    "plot_bg": "#ffffff",
    "plot_axes_bg": "#ffffff",
    "plot_text": "#1f2933",
    "plot_spine": "#6b7280",
    "plot_grid": "#cbd5e1",
    "plot_line_freq": "#0b5fff",
    "plot_line_angle": "#d84f2a",
    "plot_worst": "#dc2626",
    "plot_crosshair": "#ffffff",
}

DARK_THEME = {
    "window_bg": "#1f2430",
    "panel_bg": "#1f2430",
    "text": "#e5e7eb",
    "muted_text": "#9ca3af",
    "field_bg": "#111827",
    "field_fg": "#f9fafb",
    "field_disabled_bg": "#1f2937",
    "field_disabled_fg": "#6b7280",
    "button_bg": "#374151",
    "button_active_bg": "#4b5563",
    "selection_bg": "#1d4ed8",
    "selection_fg": "#f9fafb",
    "accent": "#60a5fa",
    "preview_bg": "#0f172a",
    "preview_border": "#475569",
    "preview_outline": "#94a3b8",
    "preview_text": "#cbd5e1",
    "preview_empty": "#94a3b8",
    "preview_layer_text": "#f8fafc",
    "preview_layer_border": "#111827",
    "layer_colors": [
        "#1d4ed8",
        "#b45309",
        "#166534",
        "#b91c1c",
        "#6d28d9",
        "#0f766e",
        "#9a3412",
        "#4d7c0f",
    ],
    "plot_bg": "#111827",
    "plot_axes_bg": "#1f2937",
    "plot_text": "#e5e7eb",
    "plot_spine": "#94a3b8",
    "plot_grid": "#475569",
    "plot_line_freq": "#60a5fa",
    "plot_line_angle": "#fb923c",
    "plot_worst": "#ef4444",
    "plot_crosshair": "#e5e7eb",
}

HEATMAP_METRIC_OPTIONS = [
    ("Metal backed loss (dB)", "metal_loss_db"),
    ("Metal phase (deg)", "metal_phase_deg"),
    ("Air backed loss (dB)", "air_loss_db"),
    ("Air phase (deg)", "air_phase_deg"),
    ("Insertion loss (dB)", "insertion_loss_db"),
    ("Insertion phase (deg)", "insertion_phase_deg"),
]
HEATMAP_METRIC_KEYS = [key for _label, key in HEATMAP_METRIC_OPTIONS]
UNCERTAINTY_VIEW_OPTIONS = [
    ("Nominal", "nominal"),
    ("Min", "min"),
    ("Max", "max"),
    ("Span (max-min)", "span"),
]
INVERSE_SCORE_MODE_OPTIONS = (
    "Worst-case mean metal loss (robust)",
    "Average mean metal loss (robust)",
)
BUILTIN_MATERIAL_PRESETS = {
    "Air (reference, low-loss)": "materials/air_reference.txt",
    "FR4 (er~4.3, tanD~0.02)": "materials/fr4_typical.txt",
    "Rogers 5880 (er~2.2)": "materials/rogers5880_typical.txt",
    "Ferrite Tile (lossy, generic)": "materials/ferrite_tile_generic.txt",
    "Carbon Loaded Foam (lossy)": "materials/carbon_loaded_foam_generic.txt",
}
_T = TypeVar("_T")


class LayerDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        initial: LayerConfig | None = None,
        presets: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Layer")
        self.resizable(False, False)
        self.result: LayerConfig | None = None
        self.transient(parent)
        self.grab_set()
        self.presets = presets or {}

        init = initial or LayerConfig(
            thickness_in=0.125,
            anisotropic=False,
            file_0deg="material.txt",
            file_90deg="",
            polarization_deg=0.0,
        )

        self.thickness_var = tk.StringVar(value=str(init.thickness_in))
        self.aniso_var = tk.BooleanVar(value=init.anisotropic)
        self.file_0deg_var = tk.StringVar(value=init.file_0deg)
        self.file_90deg_var = tk.StringVar(value=init.file_90deg)
        self.pol_var = tk.StringVar(value=str(init.polarization_deg))
        self.preset_var = tk.StringVar(value="")

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Thickness (in)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.thickness_var, width=14).grid(row=0, column=1, sticky="ew")

        ttk.Label(frm, text="Preset material").grid(row=1, column=0, sticky="w", pady=(8, 0))
        preset_values = [""] + sorted(self.presets.keys())
        self.preset_combo = ttk.Combobox(
            frm,
            textvariable=self.preset_var,
            values=preset_values,
            state="readonly",
            width=34,
        )
        self.preset_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(frm, text="Use", command=self._apply_preset).grid(row=1, column=2, padx=(6, 0), pady=(8, 0))

        ttk.Checkbutton(
            frm,
            text="Anisotropic layer (0 deg / 90 deg files)",
            variable=self.aniso_var,
            command=self._sync_state,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(frm, text="File (0 deg / isotropic)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.file_0deg_var, width=42).grid(row=3, column=1, sticky="ew", pady=(8, 0))
        ttk.Button(frm, text="Browse", command=self._browse_0deg).grid(row=3, column=2, padx=(6, 0), pady=(8, 0))

        self.lbl_90 = ttk.Label(frm, text="File (90 deg)")
        self.lbl_90.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.ent_90 = ttk.Entry(frm, textvariable=self.file_90deg_var, width=42)
        self.ent_90.grid(row=4, column=1, sticky="ew", pady=(8, 0))
        self.btn_90 = ttk.Button(frm, text="Browse", command=self._browse_90deg)
        self.btn_90.grid(row=4, column=2, padx=(6, 0), pady=(8, 0))

        self.lbl_pol = ttk.Label(frm, text="Polarization (deg)")
        self.lbl_pol.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.ent_pol = ttk.Entry(frm, textvariable=self.pol_var, width=14)
        self.ent_pol.grid(row=5, column=1, sticky="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=1)

        frm.columnconfigure(1, weight=1)
        self._sync_state()
        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self.destroy())

    def _sync_state(self) -> None:
        state = "normal" if self.aniso_var.get() else "disabled"
        self.lbl_90.configure(state=state)
        self.ent_90.configure(state=state)
        self.btn_90.configure(state=state)
        self.lbl_pol.configure(state=state)
        self.ent_pol.configure(state=state)

    def _browse_0deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 0 deg/isotropic property file")
        if p:
            self.file_0deg_var.set(p)

    def _browse_90deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 90 deg property file")
        if p:
            self.file_90deg_var.set(p)

    def _apply_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            return
        path = self.presets.get(name)
        if not path:
            return
        self.file_0deg_var.set(path)
        if self.aniso_var.get() and not self.file_90deg_var.get().strip():
            self.file_90deg_var.set(path)

    def _on_ok(self) -> None:
        try:
            thickness_in = float(self.thickness_var.get().strip())
            if thickness_in <= 0:
                raise ValueError("Thickness must be > 0.")
            anisotropic = self.aniso_var.get()
            file_0deg = self.file_0deg_var.get().strip()
            file_90deg = self.file_90deg_var.get().strip()
            polarization_deg = float(self.pol_var.get().strip()) if anisotropic else 0.0

            if not file_0deg:
                raise ValueError("0 deg/isotropic file is required.")
            if anisotropic and not file_90deg:
                raise ValueError("90 deg file is required for anisotropic layer.")

            self.result = LayerConfig(
                thickness_in=thickness_in,
                anisotropic=anisotropic,
                file_0deg=file_0deg,
                file_90deg=file_90deg,
                polarization_deg=polarization_deg,
            )
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Layer", str(exc), parent=self)


class ImpedanceGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1020x620")
        self.minsize(900, 560)

        self.layers: list[LayerConfig] = []

        self.f_start_var = tk.StringVar(value="1.0")
        self.f_stop_var = tk.StringVar(value="18.0")
        self.f_step_var = tk.StringVar(value="0.1")
        self.backing_var = tk.StringVar(value="air")
        self.skiprows_var = tk.StringVar(value="0")
        self.output_var = tk.StringVar(value="impedance_out.txt")
        self.header_var = tk.BooleanVar(value=True)
        self.angle_mode_var = tk.BooleanVar(value=False)
        self.angle_start_var = tk.StringVar(value="0.0")
        self.angle_stop_var = tk.StringVar(value="80.0")
        self.angle_step_var = tk.StringVar(value="1.0")
        self.wave_pol_var = tk.StringVar(value="HH")
        self.uncertainty_var = tk.BooleanVar(value=False)
        self.unc_t_pct_var = tk.StringVar(value="5.0")
        self.unc_eps_pct_var = tk.StringVar(value="5.0")
        self.unc_mu_pct_var = tk.StringVar(value="5.0")
        self.heatmap_metric_var = tk.StringVar(value=HEATMAP_METRIC_OPTIONS[0][0])
        self.uncertainty_view_var = tk.StringVar(value=UNCERTAINTY_VIEW_OPTIONS[0][0])
        self.metric_label_to_key = {label: key for label, key in HEATMAP_METRIC_OPTIONS}
        self.metric_key_to_label = {key: label for label, key in HEATMAP_METRIC_OPTIONS}
        self.uncertainty_view_label_to_key = {label: key for label, key in UNCERTAINTY_VIEW_OPTIONS}
        self.cbar_auto_var = tk.BooleanVar(value=True)
        self.cbar_min_var = tk.StringVar(value="")
        self.cbar_max_var = tk.StringVar(value="")
        self.slice_angle_var = tk.StringVar(value="")
        self.slice_freq_var = tk.StringVar(value="")
        self.inv_freq_mode_var = tk.StringVar(value="Band sweep")
        self.inv_freq_list_var = tk.StringVar(value="8.0, 10.0, 12.0")
        self.inv_target_start_var = tk.StringVar(value="8.0")
        self.inv_target_stop_var = tk.StringVar(value="12.0")
        self.inv_target_step_var = tk.StringVar(value="0.25")
        self.inv_angle_start_var = tk.StringVar(value="0.0")
        self.inv_angle_stop_var = tk.StringVar(value="80.0")
        self.inv_angle_step_var = tk.StringVar(value="5.0")
        self.inv_wave_pol_var = tk.StringVar(value="HH")
        self.inv_thick_min_var = tk.StringVar(value="0.03")
        self.inv_thick_max_var = tk.StringVar(value="0.30")
        self.inv_thick_steps_var = tk.StringVar(value="8")
        self.inv_material_mode_var = tk.StringVar(value="Current layer files")
        self.inv_max_evals_var = tk.StringVar(value="400")
        self.inv_top_n_var = tk.StringVar(value="10")
        self.inv_percentile_var = tk.StringVar(value="10")
        self.inv_uncertainty_var = tk.BooleanVar(value=True)
        self.inv_unc_t_pct_var = tk.StringVar(value="5.0")
        self.inv_unc_eps_pct_var = tk.StringVar(value="5.0")
        self.inv_unc_mu_pct_var = tk.StringVar(value="5.0")
        self.inv_score_mode_var = tk.StringVar(value=INVERSE_SCORE_MODE_OPTIONS[0])
        self.dashboard_expand_var = tk.BooleanVar(value=False)
        self.dashboard_expand_btn_var = tk.StringVar(value="Expand")
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.project_path: Path | None = None
        self.inverse_candidates: list[InverseCandidate] = []
        self.style = ttk.Style(self)
        self._colors = LIGHT_THEME

        # Plot objects are created in _build_ui(). Initialize here so early callbacks are safe.
        self.fig = None
        self.ax_heatmap = None
        self.ax_freq_slice = None
        self.ax_angle_slice = None
        self.canvas = None
        self.plot_frame = None
        self.heatmap_cbar = None
        self.heatmap_click_cid = None
        self.selected_angle_idx: int | None = None
        self.selected_freq_idx: int | None = None
        self.inv_results_list: tk.Listbox | None = None
        self.left_tabs = None
        self.angle_tab = None
        self.inv_tab = None
        self.inv_unc_t_entry = None
        self.inv_unc_eps_entry = None
        self.inv_unc_mu_entry = None
        self.inv_percentile_entry = None
        self.inv_target_start_entry = None
        self.inv_target_stop_entry = None
        self.inv_target_step_entry = None
        self.inv_freq_list_entry = None
        self.layer_add_btn = None
        self.layer_edit_btn = None
        self.layer_remove_btn = None
        self.layer_up_btn = None
        self.layer_down_btn = None
        self.compute_btn = None
        self.inv_run_btn = None
        self.inv_apply_btn = None
        self.status_var = tk.StringVar(value="Ready")
        self.status_progress = None
        self._task_running = False

        self.last_heatmap_results: dict[str, list[list[float]] | list[float]] | None = None
        self.last_heatmap_uncertainty_min: dict[str, list[list[float]]] | None = None
        self.last_heatmap_uncertainty_max: dict[str, list[list[float]]] | None = None
        self.inverse_plot_freqs: list[float] = []
        self.inverse_plot_samples: list[list[list[float]]] = []

        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self._build_ui()
        self._apply_theme()
        if Path("material.txt").exists():
            self.layers.append(
                LayerConfig(
                    thickness_in=0.125,
                    anisotropic=False,
                    file_0deg="material.txt",
                    file_90deg="",
                    polarization_deg=0.0,
                )
            )
            self._refresh_layers()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        split = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        split.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(split, padding=4)
        right = ttk.Frame(split, padding=4)
        split.add(left, weight=4)
        split.add(right, weight=3)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        io_frame = ttk.LabelFrame(left, text="Output")
        io_frame.grid(row=0, column=0, sticky="ew")
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Output file").grid(row=0, column=0, padx=(8, 4), pady=8)
        ttk.Entry(io_frame, textvariable=self.output_var).grid(row=0, column=1, sticky="ew", pady=8)
        ttk.Button(io_frame, text="Browse", command=self._browse_output).grid(
            row=0, column=2, padx=(6, 8), pady=8
        )

        ttk.Label(io_frame, text="Skip rows").grid(row=0, column=3, padx=(10, 4), pady=8)
        ttk.Entry(io_frame, textvariable=self.skiprows_var, width=6).grid(row=0, column=4, pady=8)
        ttk.Checkbutton(io_frame, text="Write header", variable=self.header_var).grid(
            row=0, column=5, padx=(10, 8), pady=8
        )

        self.left_tabs = ttk.Notebook(left)
        tabs = self.left_tabs
        tabs.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        angle_tab = ttk.Frame(tabs, padding=8)
        self.angle_tab = angle_tab
        angle_tab.columnconfigure(7, weight=1)
        tabs.add(angle_tab, text="Analysis")
        ttk.Label(angle_tab, text="Start (GHz)").grid(row=0, column=0, padx=(0, 4), pady=6, sticky="w")
        ttk.Entry(angle_tab, textvariable=self.f_start_var, width=10).grid(row=0, column=1, pady=6, sticky="w")
        ttk.Label(angle_tab, text="Stop (GHz)").grid(row=0, column=2, padx=(10, 4), pady=6, sticky="w")
        ttk.Entry(angle_tab, textvariable=self.f_stop_var, width=10).grid(row=0, column=3, pady=6, sticky="w")
        ttk.Label(angle_tab, text="Step (GHz)").grid(row=0, column=4, padx=(10, 4), pady=6, sticky="w")
        ttk.Entry(angle_tab, textvariable=self.f_step_var, width=10).grid(row=0, column=5, pady=6, sticky="w")
        self.backing_label = ttk.Label(angle_tab, text="Backing")
        self.backing_label.grid(row=0, column=6, padx=(10, 4), pady=6, sticky="w")
        self.backing_combo = ttk.Combobox(
            angle_tab,
            textvariable=self.backing_var,
            values=("pec", "air", "free-space"),
            state="readonly",
            width=12,
        )
        self.backing_combo.grid(row=0, column=7, sticky="w", pady=6)
        ttk.Checkbutton(
            angle_tab,
            text="Enable angle-frequency heatmap mode",
            variable=self.angle_mode_var,
            command=self._sync_mode_state,
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(2, 6))
        ttk.Label(angle_tab, text="Angle start").grid(row=2, column=0, padx=(0, 4), pady=6)
        ttk.Entry(angle_tab, textvariable=self.angle_start_var, width=10).grid(row=2, column=1, pady=6)
        ttk.Label(angle_tab, text="Angle stop").grid(row=2, column=2, padx=(10, 4), pady=6)
        ttk.Entry(angle_tab, textvariable=self.angle_stop_var, width=10).grid(row=2, column=3, pady=6)
        ttk.Label(angle_tab, text="Angle step").grid(row=2, column=4, padx=(10, 4), pady=6)
        ttk.Entry(angle_tab, textvariable=self.angle_step_var, width=10).grid(row=2, column=5, pady=6)
        ttk.Label(angle_tab, text="Wave pol").grid(row=2, column=6, padx=(10, 4), pady=6)
        ttk.Combobox(
            angle_tab,
            textvariable=self.wave_pol_var,
            values=("HH", "VV"),
            state="readonly",
            width=8,
        ).grid(row=2, column=7, sticky="w", pady=6)
        ttk.Label(
            angle_tab,
            text="Heatmap uses Frequency Sweep as Y-axis and Angle Sweep as X-axis.",
        ).grid(row=3, column=0, columnspan=8, sticky="w", pady=(0, 2))
        ttk.Checkbutton(
            angle_tab,
            text="Enable uncertainty envelope (corner sweep)",
            variable=self.uncertainty_var,
            command=self._sync_uncertainty_state,
        ).grid(row=4, column=0, columnspan=8, sticky="w", pady=(8, 0))
        self.unc_details_frame = ttk.Frame(angle_tab)
        self.unc_details_frame.grid(row=5, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        self.unc_details_frame.columnconfigure(7, weight=1)
        ttk.Label(self.unc_details_frame, text="Thickness ±%").grid(row=0, column=0, padx=(0, 4), pady=4)
        self.unc_t_entry = ttk.Entry(self.unc_details_frame, textvariable=self.unc_t_pct_var, width=10)
        self.unc_t_entry.grid(row=0, column=1, pady=4)
        ttk.Label(self.unc_details_frame, text="Eps ±%").grid(row=0, column=2, padx=(10, 4), pady=4)
        self.unc_eps_entry = ttk.Entry(self.unc_details_frame, textvariable=self.unc_eps_pct_var, width=10)
        self.unc_eps_entry.grid(row=0, column=3, pady=4)
        ttk.Label(self.unc_details_frame, text="Mu ±%").grid(row=0, column=4, padx=(10, 4), pady=4)
        self.unc_mu_entry = ttk.Entry(self.unc_details_frame, textvariable=self.unc_mu_pct_var, width=10)
        self.unc_mu_entry.grid(row=0, column=5, pady=4)
        ttk.Label(
            self.unc_details_frame,
            text="Writes nominal + min/max envelopes over ± perturbation corners.",
        ).grid(row=0, column=6, columnspan=2, sticky="w", padx=(10, 0), pady=4)
        analysis_actions = ttk.Frame(angle_tab)
        analysis_actions.grid(row=6, column=0, columnspan=8, sticky="w", pady=(10, 0))
        self.compute_btn = ttk.Button(analysis_actions, text="Compute", command=self._compute)
        self.compute_btn.grid(row=0, column=0)

        inv_tab = ttk.Frame(tabs, padding=8)
        self.inv_tab = inv_tab
        inv_tab.columnconfigure(7, weight=1)
        inv_tab.rowconfigure(8, weight=1)
        tabs.add(inv_tab, text="Inverse Design")
        ttk.Label(inv_tab, text="Freq mode").grid(row=0, column=0, padx=(0, 4), pady=6, sticky="w")
        freq_mode_combo = ttk.Combobox(
            inv_tab,
            textvariable=self.inv_freq_mode_var,
            values=("Band sweep", "Discrete list"),
            state="readonly",
            width=16,
        )
        freq_mode_combo.grid(row=0, column=1, columnspan=2, sticky="w", pady=6)
        freq_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_inverse_freq_mode_state())
        ttk.Label(inv_tab, text="Discrete f (GHz)").grid(row=0, column=3, padx=(8, 4), pady=6, sticky="e")
        self.inv_freq_list_entry = ttk.Entry(inv_tab, textvariable=self.inv_freq_list_var, width=30)
        self.inv_freq_list_entry.grid(row=0, column=4, columnspan=4, sticky="ew", pady=6)

        ttk.Label(inv_tab, text="Band start").grid(row=1, column=0, padx=(0, 4), pady=6)
        self.inv_target_start_entry = ttk.Entry(inv_tab, textvariable=self.inv_target_start_var, width=8)
        self.inv_target_start_entry.grid(row=1, column=1, pady=6)
        ttk.Label(inv_tab, text="Band stop").grid(row=1, column=2, padx=(8, 4), pady=6)
        self.inv_target_stop_entry = ttk.Entry(inv_tab, textvariable=self.inv_target_stop_var, width=8)
        self.inv_target_stop_entry.grid(row=1, column=3, pady=6)
        ttk.Label(inv_tab, text="Band step").grid(row=1, column=4, padx=(8, 4), pady=6)
        self.inv_target_step_entry = ttk.Entry(inv_tab, textvariable=self.inv_target_step_var, width=8)
        self.inv_target_step_entry.grid(row=1, column=5, pady=6)

        ttk.Label(inv_tab, text="Angle start").grid(row=2, column=0, padx=(0, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_angle_start_var, width=8).grid(row=2, column=1, pady=(2, 6))
        ttk.Label(inv_tab, text="Angle stop").grid(row=2, column=2, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_angle_stop_var, width=8).grid(row=2, column=3, pady=(2, 6))
        ttk.Label(inv_tab, text="Angle step").grid(row=2, column=4, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_angle_step_var, width=8).grid(row=2, column=5, pady=(2, 6))
        ttk.Label(inv_tab, text="Wave pol").grid(row=2, column=6, padx=(8, 4), pady=(2, 6))
        ttk.Combobox(
            inv_tab,
            textvariable=self.inv_wave_pol_var,
            values=("HH", "VV"),
            state="readonly",
            width=8,
        ).grid(row=2, column=7, sticky="w", pady=(2, 6))
        ttk.Label(inv_tab, text="Tip: set Angle start == Angle stop to analyze a single angle.").grid(
            row=3,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(0, 6),
        )

        ttk.Label(inv_tab, text="t_min").grid(row=4, column=0, padx=(0, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_thick_min_var, width=8).grid(row=4, column=1, pady=(2, 6))
        ttk.Label(inv_tab, text="t_max").grid(row=4, column=2, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_thick_max_var, width=8).grid(row=4, column=3, pady=(2, 6))
        ttk.Label(inv_tab, text="t_steps").grid(row=4, column=4, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_thick_steps_var, width=8).grid(row=4, column=5, pady=(2, 6))

        ttk.Label(inv_tab, text="Material mode").grid(row=5, column=0, padx=(0, 4), pady=(2, 6))
        ttk.Combobox(
            inv_tab,
            textvariable=self.inv_material_mode_var,
            values=("Current layer files", "Search .txt materials in folder"),
            state="readonly",
            width=28,
        ).grid(row=5, column=1, columnspan=3, sticky="w", pady=(2, 6))
        ttk.Label(inv_tab, text="Max evals").grid(row=5, column=4, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_max_evals_var, width=8).grid(row=5, column=5, pady=(2, 6))
        ttk.Label(inv_tab, text="Top N").grid(row=5, column=6, padx=(8, 4), pady=(2, 6))
        ttk.Entry(inv_tab, textvariable=self.inv_top_n_var, width=8).grid(row=5, column=7, pady=(2, 6), sticky="w")

        ttk.Checkbutton(
            inv_tab,
            text="Enable robust scoring (uncertainty corners)",
            variable=self.inv_uncertainty_var,
            command=self._sync_inverse_uncertainty_state,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 6))
        ttk.Label(inv_tab, text="T ±%").grid(row=6, column=4, padx=(4, 4), pady=(2, 6))
        self.inv_unc_t_entry = ttk.Entry(inv_tab, textvariable=self.inv_unc_t_pct_var, width=7)
        self.inv_unc_t_entry.grid(row=6, column=5, pady=(2, 6), sticky="w")
        ttk.Label(inv_tab, text="Eps ±%").grid(row=6, column=6, padx=(4, 4), pady=(2, 6))
        self.inv_unc_eps_entry = ttk.Entry(inv_tab, textvariable=self.inv_unc_eps_pct_var, width=7)
        self.inv_unc_eps_entry.grid(row=6, column=7, pady=(2, 6), sticky="w")
        ttk.Label(inv_tab, text="Mu ±%").grid(row=7, column=0, padx=(0, 4), pady=(0, 6), sticky="w")
        self.inv_unc_mu_entry = ttk.Entry(inv_tab, textvariable=self.inv_unc_mu_pct_var, width=7)
        self.inv_unc_mu_entry.grid(row=7, column=1, pady=(0, 6), sticky="w")
        ttk.Label(inv_tab, text="Score mode").grid(row=7, column=2, padx=(8, 4), pady=(0, 6))
        ttk.Combobox(
            inv_tab,
            textvariable=self.inv_score_mode_var,
            values=INVERSE_SCORE_MODE_OPTIONS,
            state="readonly",
            width=34,
        ).grid(row=7, column=3, columnspan=5, sticky="w", pady=(0, 6))

        results_frame = ttk.Frame(inv_tab)
        results_frame.grid(row=8, column=0, columnspan=8, sticky="nsew", pady=(2, 6))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.inv_results_list = tk.Listbox(results_frame, height=7, exportselection=False)
        self.inv_results_list.grid(row=0, column=0, sticky="nsew")
        self.inv_results_list.bind("<<ListboxSelect>>", lambda _e: self._update_plot())
        inv_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.inv_results_list.yview)
        inv_scroll.grid(row=0, column=1, sticky="ns")
        self.inv_results_list.configure(yscrollcommand=inv_scroll.set)

        inv_actions = ttk.Frame(inv_tab)
        inv_actions.grid(row=9, column=0, columnspan=8, sticky="w")
        self.inv_run_btn = ttk.Button(inv_actions, text="Run Inverse Design", command=self._run_inverse_design)
        self.inv_run_btn.grid(row=0, column=0)
        self.inv_apply_btn = ttk.Button(inv_actions, text="Apply Selected", command=self._apply_inverse_candidate)
        self.inv_apply_btn.grid(row=0, column=1, padx=(8, 0))
        ttk.Label(inv_actions, text="Percentile").grid(row=0, column=2, padx=(16, 4))
        self.inv_percentile_entry = ttk.Entry(inv_actions, textvariable=self.inv_percentile_var, width=6)
        self.inv_percentile_entry.grid(row=0, column=3, sticky="w")
        ttk.Label(inv_actions, text="%").grid(row=0, column=4, padx=(2, 0))
        self.inv_percentile_entry.bind("<Return>", lambda _e: self._on_inverse_percentile_changed())
        self.inv_percentile_entry.bind("<FocusOut>", lambda _e: self._on_inverse_percentile_changed())

        stats_tab = ttk.Frame(tabs, padding=8)
        stats_tab.columnconfigure(0, weight=1)
        stats_tab.rowconfigure(1, weight=1)
        tabs.add(stats_tab, text="Quick Stats")
        dash_toolbar = ttk.Frame(stats_tab)
        dash_toolbar.grid(row=0, column=0, sticky="e", pady=(0, 6))
        ttk.Button(
            dash_toolbar,
            textvariable=self.dashboard_expand_btn_var,
            command=self._toggle_dashboard_expand,
            width=9,
        ).grid(row=0, column=0)
        self.dashboard_text = tk.Text(stats_tab, height=12, wrap="word")
        self.dashboard_text.grid(row=1, column=0, sticky="nsew")
        self.dashboard_text.configure(state="disabled")

        layers_frame = ttk.LabelFrame(left, text="Layers (top to bottom)")
        layers_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        layers_frame.columnconfigure(0, weight=1)
        layers_frame.columnconfigure(2, weight=1)
        layers_frame.rowconfigure(0, weight=1)

        self.layer_list = tk.Listbox(layers_frame, height=12)
        self.layer_list.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        scroll = ttk.Scrollbar(layers_frame, orient="vertical", command=self.layer_list.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=8)
        self.layer_list.configure(yscrollcommand=scroll.set)

        preview = ttk.Frame(layers_frame)
        preview.grid(row=0, column=2, sticky="nsew", padx=8, pady=8)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        ttk.Label(preview, text="Visual stack (real-time)").grid(row=0, column=0, sticky="w")
        self.layer_preview = tk.Canvas(
            preview,
            width=250,
            height=250,
            highlightthickness=1,
            highlightbackground=LIGHT_THEME["preview_border"],
            bg=LIGHT_THEME["preview_bg"],
        )
        self.layer_preview.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.layer_preview.bind("<Configure>", self._on_layer_preview_configure)

        btns = ttk.Frame(layers_frame)
        btns.grid(row=0, column=3, sticky="ns", padx=8, pady=8)
        self.layer_add_btn = ttk.Button(btns, text="Add", width=12, command=self._add_layer)
        self.layer_add_btn.grid(row=0, column=0, pady=(0, 6))
        self.layer_edit_btn = ttk.Button(btns, text="Edit", width=12, command=self._edit_layer)
        self.layer_edit_btn.grid(row=1, column=0, pady=6)
        self.layer_remove_btn = ttk.Button(btns, text="Remove", width=12, command=self._remove_layer)
        self.layer_remove_btn.grid(row=2, column=0, pady=6)
        self.layer_up_btn = ttk.Button(btns, text="Move Up", width=12, command=self._move_up)
        self.layer_up_btn.grid(row=3, column=0, pady=6)
        self.layer_down_btn = ttk.Button(btns, text="Move Down", width=12, command=self._move_down)
        self.layer_down_btn.grid(row=4, column=0, pady=6)

        action = ttk.Frame(left)
        action.grid(row=3, column=0, sticky="e", pady=(12, 0))
        ttk.Checkbutton(
            action,
            text="Dark mode",
            variable=self.dark_mode_var,
            command=self._on_theme_toggle,
        ).grid(row=0, column=0, padx=(0, 12))
        ttk.Button(action, text="Load Project", command=self._load_project).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(action, text="Save Project", command=self._save_project).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(action, text="About", command=self._show_about).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(action, textvariable=self.status_var).grid(row=0, column=4, padx=(10, 4))
        self.status_progress = ttk.Progressbar(action, mode="indeterminate", length=120)
        self.status_progress.grid(row=0, column=5, padx=(0, 2))
        self.status_progress.grid_remove()

        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        plot_opts = ttk.LabelFrame(right, text="Heatmap Controls")
        plot_opts.grid(row=0, column=0, sticky="ew")
        plot_opts.columnconfigure(1, weight=1)
        plot_opts.columnconfigure(3, weight=1)
        ttk.Label(plot_opts, text="Metric").grid(row=0, column=0, padx=(8, 4), pady=8, sticky="w")
        metric_combo = ttk.Combobox(
            plot_opts,
            textvariable=self.heatmap_metric_var,
            values=[label for label, _key in HEATMAP_METRIC_OPTIONS],
            state="readonly",
        )
        metric_combo.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="ew")
        metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_plot())
        ttk.Label(plot_opts, text="Uncertainty view").grid(row=0, column=2, padx=(0, 4), pady=8, sticky="w")
        unc_view_combo = ttk.Combobox(
            plot_opts,
            textvariable=self.uncertainty_view_var,
            values=[label for label, _key in UNCERTAINTY_VIEW_OPTIONS],
            state="readonly",
            width=14,
        )
        unc_view_combo.grid(row=0, column=3, padx=(0, 8), pady=8, sticky="ew")
        unc_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_plot())
        ttk.Button(plot_opts, text="Update Plot", command=self._update_plot).grid(
            row=0, column=4, padx=(0, 8), pady=8
        )

        ttk.Label(plot_opts, text="Angle slice (deg)").grid(row=1, column=0, padx=(8, 4), pady=8, sticky="w")
        self.slice_angle_entry = ttk.Entry(plot_opts, textvariable=self.slice_angle_var, width=10)
        self.slice_angle_entry.grid(row=1, column=1, padx=(0, 8), pady=8, sticky="w")
        ttk.Label(plot_opts, text="Freq slice (GHz)").grid(row=1, column=2, padx=(0, 4), pady=8, sticky="w")
        self.slice_freq_entry = ttk.Entry(plot_opts, textvariable=self.slice_freq_var, width=10)
        self.slice_freq_entry.grid(row=1, column=3, padx=(0, 8), pady=8, sticky="w")
        ttk.Checkbutton(
            plot_opts,
            text="Auto color scale",
            variable=self.cbar_auto_var,
            command=self._sync_cbar_state,
        ).grid(row=1, column=4, padx=(8, 4), pady=8, sticky="w")
        ttk.Label(plot_opts, text="Min").grid(row=1, column=5, padx=(0, 4), pady=8, sticky="e")
        self.cbar_min_entry = ttk.Entry(plot_opts, textvariable=self.cbar_min_var, width=10)
        self.cbar_min_entry.grid(row=1, column=6, padx=(0, 8), pady=8, sticky="w")
        ttk.Label(plot_opts, text="Max").grid(row=1, column=7, padx=(0, 4), pady=8, sticky="e")
        self.cbar_max_entry = ttk.Entry(plot_opts, textvariable=self.cbar_max_var, width=10)
        self.cbar_max_entry.grid(row=1, column=8, padx=(0, 8), pady=8, sticky="w")
        self.slice_angle_entry.bind("<Return>", lambda _e: self._apply_manual_slices())
        self.slice_freq_entry.bind("<Return>", lambda _e: self._apply_manual_slices())
        self.cbar_min_entry.bind("<Return>", lambda _e: self._update_plot())
        self.cbar_max_entry.bind("<Return>", lambda _e: self._update_plot())
        ttk.Button(plot_opts, text="Save Plot", command=self._save_plot).grid(
            row=1, column=9, padx=(0, 8), pady=8
        )
        ttk.Button(plot_opts, text="Save Heatmap Only", command=self._save_heatmap_only).grid(
            row=1, column=10, padx=(0, 8), pady=8
        )
        self._sync_cbar_state()

        self.plot_frame = ttk.LabelFrame(right, text="Heatmap")
        self.plot_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        if MPL_AVAILABLE:
            self.fig = Figure(figsize=(6.8, 5.2), dpi=100)
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2.6, 1.5], hspace=0.42)
            self.ax_heatmap = self.fig.add_subplot(gs[0, :])
            self.ax_freq_slice = self.fig.add_subplot(gs[1, 0])
            self.ax_angle_slice = self.fig.add_subplot(gs[1, 1])
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.heatmap_cbar = None
            self.heatmap_click_cid = self.canvas.mpl_connect("button_press_event", self._on_plot_click)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self._draw_plot_placeholder("Run heatmap mode to populate plot.")
        else:
            self.fig = None
            self.ax_heatmap = None
            self.ax_freq_slice = None
            self.ax_angle_slice = None
            self.canvas = None
            self.heatmap_cbar = None
            ttk.Label(
                self.plot_frame,
                text="Matplotlib not available. Install matplotlib to enable plotting.",
            ).grid(row=0, column=0, padx=12, pady=12, sticky="nw")
        self._set_dashboard_text("Run a compute to populate dashboard and interactive slices.")
        self._sync_mode_state()
        self._sync_uncertainty_state()
        self._sync_inverse_freq_mode_state()
        self._sync_inverse_uncertainty_state()
        tabs.bind("<<NotebookTabChanged>>", self._on_left_tab_changed)
        self._draw_layer_preview()

    def _on_theme_toggle(self) -> None:
        self._apply_theme()

    def _theme_colors(self) -> dict[str, object]:
        return DARK_THEME if self.dark_mode_var.get() else LIGHT_THEME

    def _safe_style_configure(self, style_name: str, **kwargs: object) -> None:
        try:
            self.style.configure(style_name, **kwargs)
        except tk.TclError:
            pass

    def _safe_style_map(self, style_name: str, **kwargs: object) -> None:
        try:
            self.style.map(style_name, **kwargs)
        except tk.TclError:
            pass

    def _style_plot_axis(self, axis: object) -> None:
        style_axis(axis, self._colors)

    def _apply_theme(self) -> None:
        colors = self._theme_colors()
        self._colors = colors
        self.configure(bg=colors["window_bg"])

        self._safe_style_configure(
            ".",
            background=colors["panel_bg"],
            foreground=colors["text"],
            fieldbackground=colors["field_bg"],
        )
        self._safe_style_configure("TFrame", background=colors["panel_bg"])
        self._safe_style_configure("TLabel", background=colors["panel_bg"], foreground=colors["text"])
        self._safe_style_configure("TLabelframe", background=colors["panel_bg"], foreground=colors["text"])
        self._safe_style_configure("TLabelframe.Label", background=colors["panel_bg"], foreground=colors["text"])
        self._safe_style_configure("TButton", background=colors["button_bg"], foreground=colors["text"])
        self._safe_style_map(
            "TButton",
            background=[("active", colors["button_active_bg"]), ("disabled", colors["field_disabled_bg"])],
            foreground=[("disabled", colors["field_disabled_fg"])],
        )
        self._safe_style_configure("TCheckbutton", background=colors["panel_bg"], foreground=colors["text"])
        self._safe_style_map("TCheckbutton", foreground=[("disabled", colors["field_disabled_fg"])])
        self._safe_style_configure(
            "TEntry",
            fieldbackground=colors["field_bg"],
            foreground=colors["field_fg"],
        )
        self._safe_style_map(
            "TEntry",
            fieldbackground=[("disabled", colors["field_disabled_bg"])],
            foreground=[("disabled", colors["field_disabled_fg"])],
        )
        self._safe_style_configure(
            "TCombobox",
            fieldbackground=colors["field_bg"],
            background=colors["field_bg"],
            foreground=colors["field_fg"],
            arrowcolor=colors["text"],
        )
        self._safe_style_map(
            "TCombobox",
            fieldbackground=[
                ("readonly", colors["field_bg"]),
                ("disabled", colors["field_disabled_bg"]),
            ],
            foreground=[
                ("readonly", colors["field_fg"]),
                ("disabled", colors["field_disabled_fg"]),
            ],
            selectbackground=[("readonly", colors["selection_bg"])],
            selectforeground=[("readonly", colors["selection_fg"])],
        )
        self._safe_style_configure("TNotebook", background=colors["panel_bg"])
        self._safe_style_configure("TNotebook.Tab", background=colors["button_bg"], foreground=colors["text"])
        self._safe_style_map(
            "TNotebook.Tab",
            background=[("selected", colors["field_bg"]), ("active", colors["button_active_bg"])],
            foreground=[("selected", colors["field_fg"])],
        )
        self._safe_style_configure(
            "Horizontal.TProgressbar",
            background=colors["accent"],
            troughcolor=colors["field_disabled_bg"],
        )
        self._safe_style_configure(
            "TScrollbar",
            background=colors["button_bg"],
            troughcolor=colors["panel_bg"],
            arrowcolor=colors["text"],
        )

        self.option_add("*TCombobox*Listbox.background", colors["field_bg"])
        self.option_add("*TCombobox*Listbox.foreground", colors["field_fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", colors["selection_bg"])
        self.option_add("*TCombobox*Listbox.selectForeground", colors["selection_fg"])

        self.layer_list.configure(
            bg=colors["field_bg"],
            fg=colors["field_fg"],
            selectbackground=colors["selection_bg"],
            selectforeground=colors["selection_fg"],
            highlightbackground=colors["preview_border"],
            highlightcolor=colors["accent"],
        )
        if self.inv_results_list is not None:
            self.inv_results_list.configure(
                bg=colors["field_bg"],
                fg=colors["field_fg"],
                selectbackground=colors["selection_bg"],
                selectforeground=colors["selection_fg"],
                highlightbackground=colors["preview_border"],
                highlightcolor=colors["accent"],
            )
        self.dashboard_text.configure(
            bg=colors["field_bg"],
            fg=colors["field_fg"],
            insertbackground=colors["field_fg"],
            selectbackground=colors["selection_bg"],
            selectforeground=colors["selection_fg"],
            highlightbackground=colors["preview_border"],
            highlightcolor=colors["accent"],
        )
        self.layer_preview.configure(
            bg=colors["preview_bg"],
            highlightbackground=colors["preview_border"],
        )
        self._draw_layer_preview()

        if self.canvas is not None and self.fig is not None:
            self.fig.patch.set_facecolor(colors["plot_bg"])
            self.canvas.get_tk_widget().configure(bg=colors["panel_bg"])
            self._update_plot()

    def _browse_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if p:
            self.output_var.set(p)

    def _coerce_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _collect_project_state(self) -> dict[str, object]:
        controls: dict[str, object] = {
            "f_start": self.f_start_var.get(),
            "f_stop": self.f_stop_var.get(),
            "f_step": self.f_step_var.get(),
            "backing": self.backing_var.get(),
            "skiprows": self.skiprows_var.get(),
            "output": self.output_var.get(),
            "header": self.header_var.get(),
            "angle_mode": self.angle_mode_var.get(),
            "angle_start": self.angle_start_var.get(),
            "angle_stop": self.angle_stop_var.get(),
            "angle_step": self.angle_step_var.get(),
            "wave_pol": self.wave_pol_var.get(),
            "uncertainty": self.uncertainty_var.get(),
            "unc_t_pct": self.unc_t_pct_var.get(),
            "unc_eps_pct": self.unc_eps_pct_var.get(),
            "unc_mu_pct": self.unc_mu_pct_var.get(),
            "heatmap_metric": self.heatmap_metric_var.get(),
            "uncertainty_view": self.uncertainty_view_var.get(),
            "cbar_auto": self.cbar_auto_var.get(),
            "cbar_min": self.cbar_min_var.get(),
            "cbar_max": self.cbar_max_var.get(),
            "slice_angle": self.slice_angle_var.get(),
            "slice_freq": self.slice_freq_var.get(),
            "inv_freq_mode": self.inv_freq_mode_var.get(),
            "inv_freq_list": self.inv_freq_list_var.get(),
            "inv_target_start": self.inv_target_start_var.get(),
            "inv_target_stop": self.inv_target_stop_var.get(),
            "inv_target_step": self.inv_target_step_var.get(),
            "inv_angle_start": self.inv_angle_start_var.get(),
            "inv_angle_stop": self.inv_angle_stop_var.get(),
            "inv_angle_step": self.inv_angle_step_var.get(),
            "inv_wave_pol": self.inv_wave_pol_var.get(),
            "inv_thick_min": self.inv_thick_min_var.get(),
            "inv_thick_max": self.inv_thick_max_var.get(),
            "inv_thick_steps": self.inv_thick_steps_var.get(),
            "inv_material_mode": self.inv_material_mode_var.get(),
            "inv_max_evals": self.inv_max_evals_var.get(),
            "inv_top_n": self.inv_top_n_var.get(),
            "inv_percentile": self.inv_percentile_var.get(),
            "inv_uncertainty": self.inv_uncertainty_var.get(),
            "inv_unc_t_pct": self.inv_unc_t_pct_var.get(),
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var.get(),
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var.get(),
            "inv_score_mode": self.inv_score_mode_var.get(),
            "dashboard_expand": self.dashboard_expand_var.get(),
            "dark_mode": self.dark_mode_var.get(),
        }
        return {
            "layers": [layer_config_to_dict(layer) for layer in self.layers],
            "controls": controls,
        }

    def _apply_project_state(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Project state must be an object.")
        layers_data = state.get("layers", [])
        controls = state.get("controls", {})
        if not isinstance(layers_data, list):
            raise ValueError("Project layers must be a list.")
        if not isinstance(controls, dict):
            raise ValueError("Project controls must be an object.")

        loaded_layers: list[LayerConfig] = []
        for idx, raw_layer in enumerate(layers_data, start=1):
            if not isinstance(raw_layer, dict):
                raise ValueError(f"Layer {idx}: expected an object.")
            loaded_layers.append(layer_config_from_dict(raw_layer, idx))

        str_vars: dict[str, tk.StringVar] = {
            "f_start": self.f_start_var,
            "f_stop": self.f_stop_var,
            "f_step": self.f_step_var,
            "backing": self.backing_var,
            "skiprows": self.skiprows_var,
            "output": self.output_var,
            "angle_start": self.angle_start_var,
            "angle_stop": self.angle_stop_var,
            "angle_step": self.angle_step_var,
            "wave_pol": self.wave_pol_var,
            "unc_t_pct": self.unc_t_pct_var,
            "unc_eps_pct": self.unc_eps_pct_var,
            "unc_mu_pct": self.unc_mu_pct_var,
            "heatmap_metric": self.heatmap_metric_var,
            "uncertainty_view": self.uncertainty_view_var,
            "cbar_min": self.cbar_min_var,
            "cbar_max": self.cbar_max_var,
            "slice_angle": self.slice_angle_var,
            "slice_freq": self.slice_freq_var,
            "inv_freq_mode": self.inv_freq_mode_var,
            "inv_freq_list": self.inv_freq_list_var,
            "inv_target_start": self.inv_target_start_var,
            "inv_target_stop": self.inv_target_stop_var,
            "inv_target_step": self.inv_target_step_var,
            "inv_angle_start": self.inv_angle_start_var,
            "inv_angle_stop": self.inv_angle_stop_var,
            "inv_angle_step": self.inv_angle_step_var,
            "inv_wave_pol": self.inv_wave_pol_var,
            "inv_thick_min": self.inv_thick_min_var,
            "inv_thick_max": self.inv_thick_max_var,
            "inv_thick_steps": self.inv_thick_steps_var,
            "inv_material_mode": self.inv_material_mode_var,
            "inv_max_evals": self.inv_max_evals_var,
            "inv_top_n": self.inv_top_n_var,
            "inv_percentile": self.inv_percentile_var,
            "inv_unc_t_pct": self.inv_unc_t_pct_var,
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var,
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var,
            "inv_score_mode": self.inv_score_mode_var,
        }
        bool_vars: dict[str, tk.BooleanVar] = {
            "header": self.header_var,
            "angle_mode": self.angle_mode_var,
            "uncertainty": self.uncertainty_var,
            "cbar_auto": self.cbar_auto_var,
            "inv_uncertainty": self.inv_uncertainty_var,
            "dashboard_expand": self.dashboard_expand_var,
            "dark_mode": self.dark_mode_var,
        }

        for key, var in str_vars.items():
            if key in controls:
                var.set(str(controls[key]))
        for key, var in bool_vars.items():
            if key in controls:
                var.set(self._coerce_bool(controls[key]))

        self.layers = loaded_layers
        self._refresh_layers()
        self._sync_mode_state()
        self._sync_uncertainty_state()
        self._sync_inverse_freq_mode_state()
        self._sync_inverse_uncertainty_state()
        self._sync_cbar_state()

        expanded = self.dashboard_expand_var.get()
        self.dashboard_expand_btn_var.set("Collapse" if expanded else "Expand")
        self.dashboard_text.configure(height=14 if expanded else 6)

        self._apply_theme()
        self.last_heatmap_results = None
        self.last_heatmap_uncertainty_min = None
        self.last_heatmap_uncertainty_max = None
        self.inverse_plot_freqs = []
        self.inverse_plot_samples = []
        self.selected_angle_idx = None
        self.selected_freq_idx = None
        self.inverse_candidates = []
        self._refresh_inverse_results_list()
        self._set_dashboard_text("Project loaded. Run compute to generate fresh results.")
        self._update_plot()

    def _save_project(self) -> None:
        try:
            if self.project_path is None:
                path_str = filedialog.asksaveasfilename(
                    title="Save project",
                    defaultextension=".json",
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                )
                if not path_str:
                    return
                self.project_path = Path(path_str)
            save_project_file(self.project_path, self._collect_project_state())
            messagebox.showinfo("Project", f"Saved project to:\n{self.project_path}")
        except Exception as exc:
            messagebox.showerror("Project Save Error", str(exc))

    def _load_project(self) -> None:
        try:
            path_str = filedialog.askopenfilename(
                title="Load project",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not path_str:
                return
            path = Path(path_str)
            state = load_project_file(path)
            self._apply_project_state(state)
            self.project_path = path
            messagebox.showinfo("Project", f"Loaded project from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Project Load Error", str(exc))

    def _sync_cbar_state(self) -> None:
        state = "disabled" if self.cbar_auto_var.get() else "normal"
        self.cbar_min_entry.configure(state=state)
        self.cbar_max_entry.configure(state=state)
        if self.canvas is not None:
            self._update_plot()

    def _sync_mode_state(self) -> None:
        if self.angle_mode_var.get():
            self.backing_combo.configure(state="disabled")
            self.backing_label.state(["disabled"])
            return
        self.backing_combo.configure(state="readonly")
        self.backing_label.state(["!disabled"])

    def _sync_uncertainty_state(self) -> None:
        if self.uncertainty_var.get():
            self.unc_details_frame.grid()
            self.unc_t_entry.configure(state="normal")
            self.unc_eps_entry.configure(state="normal")
            self.unc_mu_entry.configure(state="normal")
            return
        self.unc_details_frame.grid_remove()
        self.unc_t_entry.configure(state="disabled")
        self.unc_eps_entry.configure(state="disabled")
        self.unc_mu_entry.configure(state="disabled")

    def _sync_inverse_uncertainty_state(self) -> None:
        state = "normal" if self.inv_uncertainty_var.get() else "disabled"
        if self.inv_unc_t_entry is not None:
            self.inv_unc_t_entry.configure(state=state)
        if self.inv_unc_eps_entry is not None:
            self.inv_unc_eps_entry.configure(state=state)
        if self.inv_unc_mu_entry is not None:
            self.inv_unc_mu_entry.configure(state=state)

    def _sync_inverse_freq_mode_state(self) -> None:
        mode = self.inv_freq_mode_var.get().strip().lower()
        band_enabled = mode.startswith("band")
        band_state = "normal" if band_enabled else "disabled"
        discrete_state = "disabled" if band_enabled else "normal"
        for entry in (
            self.inv_target_start_entry,
            self.inv_target_stop_entry,
            self.inv_target_step_entry,
        ):
            if entry is not None:
                entry.configure(state=band_state)
        if self.inv_freq_list_entry is not None:
            self.inv_freq_list_entry.configure(state=discrete_state)

    def _on_inverse_percentile_changed(self) -> None:
        text = self.inv_percentile_var.get().strip()
        if not text:
            self.inv_percentile_var.set("10")
            self._update_plot()
            return
        try:
            p = float(text)
        except Exception:
            messagebox.showerror("Inverse Plot", "Percentile must be a number between 0 and 100.")
            return
        if p < 0.0 or p > 100.0:
            messagebox.showerror("Inverse Plot", "Percentile must be between 0 and 100.")
            return
        self.inv_percentile_var.set(f"{p:g}")
        self._update_plot()

    def _current_inverse_percentile(self) -> float:
        text = self.inv_percentile_var.get().strip()
        try:
            p = float(text)
        except Exception:
            return 10.0
        return max(0.0, min(100.0, p))

    def _toggle_dashboard_expand(self) -> None:
        expanded = not self.dashboard_expand_var.get()
        self.dashboard_expand_var.set(expanded)
        self.dashboard_expand_btn_var.set("Collapse" if expanded else "Expand")
        self.dashboard_text.configure(height=14 if expanded else 6)

    def _read_uncertainty_config(self) -> UncertaintyConfig:
        if not self.uncertainty_var.get():
            return UncertaintyConfig(enabled=False, thickness_pct=0.0, eps_pct=0.0, mu_pct=0.0)

        thickness_pct = float(self.unc_t_pct_var.get().strip())
        eps_pct = float(self.unc_eps_pct_var.get().strip())
        mu_pct = float(self.unc_mu_pct_var.get().strip())
        if thickness_pct < 0 or eps_pct < 0 or mu_pct < 0:
            raise ValueError("Uncertainty percentages must be >= 0.")
        return UncertaintyConfig(
            enabled=True,
            thickness_pct=thickness_pct,
            eps_pct=eps_pct,
            mu_pct=mu_pct,
        )

    def _save_plot(self) -> None:
        if not MPL_AVAILABLE or self.fig is None:
            messagebox.showerror("Plot", "Matplotlib is not available.")
            return
        p = filedialog.asksaveasfilename(
            title="Save plot image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")],
        )
        if not p:
            return
        try:
            self.fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Plot", f"Saved plot to:\n{p}")
        except Exception as exc:
            messagebox.showerror("Plot", str(exc))

    def _save_heatmap_only(self) -> None:
        if not MPL_AVAILABLE:
            messagebox.showerror("Heatmap", "Matplotlib is not available.")
            return
        if self.last_heatmap_results is None:
            messagebox.showerror("Heatmap", "No heatmap data to save. Run Analysis compute first.")
            return

        selected = self._get_selected_metric_grid()
        if selected is None:
            messagebox.showerror("Heatmap", "Select a valid heatmap metric first.")
            return
        metric_label, metric_key, z = selected

        try:
            cmin, cmax = self._get_color_limits()
        except Exception as exc:
            messagebox.showerror("Heatmap", str(exc))
            return

        p = filedialog.asksaveasfilename(
            title="Save heatmap image (no crosshairs)",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")],
        )
        if not p:
            return

        try:
            angles = self.last_heatmap_results["angle_deg"]
            freqs = self.last_heatmap_results["freq_ghz"]
            cmap = "magma" if "[span]" in metric_label else ("twilight" if "phase" in metric_key else "viridis")

            fig = Figure(figsize=(7.0, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=(angles[0], angles[-1], freqs[0], freqs[-1]),
                cmap=cmap,
                vmin=cmin,
                vmax=cmax,
            )
            ax.set_title(f"{metric_label} vs Angle/Frequency")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Frequency (GHz)")
            ax.grid(False)
            self._style_plot_axis(ax)
            cbar = fig.colorbar(im, ax=ax)
            if cmin is None or cmax is None:
                cbar.set_label(metric_label)
            else:
                cbar.set_label(f"{metric_label} [{cmin:g}, {cmax:g}]")
            style_colorbar(cbar, self._colors)
            fig.patch.set_facecolor(self._colors["plot_bg"])
            fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Heatmap", f"Saved heatmap to:\n{p}")
        except Exception as exc:
            messagebox.showerror("Heatmap", str(exc))

    def _show_about(self) -> None:
        messagebox.showinfo(f"About {APP_ACRONYM}", ABOUT_TEXT)

    def _get_color_limits(self) -> tuple[float | None, float | None]:
        if self.cbar_auto_var.get():
            return None, None
        cmin_text = self.cbar_min_var.get().strip()
        cmax_text = self.cbar_max_var.get().strip()
        if not cmin_text or not cmax_text:
            raise ValueError("Set both colorbar Min and Max, or enable Auto color scale.")
        cmin = float(cmin_text)
        cmax = float(cmax_text)
        if cmax <= cmin:
            raise ValueError("Colorbar Max must be greater than Min.")
        return cmin, cmax

    def _stats(self, values: list[float]) -> tuple[float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan")
        if NUMPY_AVAILABLE:
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.min()), float(arr.max())
        mean = sum(values) / len(values)
        return mean, min(values), max(values)

    def _max_contiguous_bandwidth(
        self,
        freqs: list[float],
        values: list[float],
        threshold: float,
    ) -> float:
        if len(freqs) < 2:
            return 0.0
        best = 0.0
        run_start: int | None = None
        for i, v in enumerate(values):
            if v >= threshold:
                if run_start is None:
                    run_start = i
            elif run_start is not None:
                best = max(best, freqs[i - 1] - freqs[run_start])
                run_start = None
        if run_start is not None:
            best = max(best, freqs[-1] - freqs[run_start])
        return max(best, 0.0)

    def _summarize_angle_run(
        self,
        out: dict[str, list[list[float]] | list[float]],
        wave_pol: str,
        uncertainty_enabled: bool,
    ) -> str:
        freqs = out["freq_ghz"]
        angles = out["angle_deg"]
        metal = out["metal_loss_db"]
        air = out["air_loss_db"]
        ins = out["insertion_loss_db"]
        phase = out["metal_phase_deg"]

        all_metal = [v for row in metal for v in row]
        all_air = [v for row in air for v in row]
        all_ins = [v for row in ins for v in row]
        all_phase = [v for row in phase for v in row]
        metal_mean, metal_min, metal_max = self._stats(all_metal)
        air_mean, air_min, air_max = self._stats(all_air)
        ins_mean, ins_min, ins_max = self._stats(all_ins)
        phase_mean, phase_min, phase_max = self._stats(all_phase)

        band_threshold = 10.0
        best_bw = 0.0
        best_bw_angle = angles[0]
        for j, a in enumerate(angles):
            row = [metal[i][j] for i in range(len(freqs))]
            bw = self._max_contiguous_bandwidth(freqs, row, band_threshold)
            if bw > best_bw:
                best_bw = bw
                best_bw_angle = a

        best_angle = angles[0]
        best_angle_score = float("-inf")
        for j, a in enumerate(angles):
            row = [metal[i][j] for i in range(len(freqs))]
            score, _mn, _mx = self._stats(row)
            if score > best_angle_score:
                best_angle_score = score
                best_angle = a

        unc_state = "ON" if uncertainty_enabled else "OFF"
        return (
            f"Mode: angle-frequency heatmap ({wave_pol.upper()}) | Uncertainty: {unc_state}\n"
            f"Grid: {len(freqs)} freq x {len(angles)} angle points\n"
            f"Metal loss dB mean/min/max: {metal_mean:.3f} / {metal_min:.3f} / {metal_max:.3f}\n"
            f"Air loss dB mean/min/max: {air_mean:.3f} / {air_min:.3f} / {air_max:.3f}\n"
            f"Insertion loss dB mean/min/max: {ins_mean:.3f} / {ins_min:.3f} / {ins_max:.3f}\n"
            f"Metal phase deg mean/min/max: {phase_mean:.3f} / {phase_min:.3f} / {phase_max:.3f}\n"
            f"Best average metal loss angle: {best_angle:.2f} deg ({best_angle_score:.3f} dB)\n"
            f"Max contiguous bandwidth with metal loss >= {band_threshold:.0f} dB: "
            f"{best_bw:.3f} GHz @ {best_bw_angle:.2f} deg"
        )

    def _summarize_frequency_run(
        self,
        sweep: list[float],
        loaded_layers: list[LoadedLayer],
        wave_pol: str,
        uncertainty_enabled: bool,
        backing: str,
    ) -> str:
        metrics = compute_angle_metrics_many(sweep, 0.0, loaded_layers, wave_pol)
        metal = metrics["metal_loss_db"]
        air = metrics["air_loss_db"]
        ins = metrics["insertion_loss_db"]
        phase = metrics["metal_phase_deg"]
        metal_mean, metal_min, metal_max = self._stats(metal)
        air_mean, air_min, air_max = self._stats(air)
        ins_mean, ins_min, ins_max = self._stats(ins)
        phase_mean, phase_min, phase_max = self._stats(phase)
        bw10 = self._max_contiguous_bandwidth(sweep, metal, 10.0)
        unc_state = "ON" if uncertainty_enabled else "OFF"
        return (
            f"Mode: frequency sweep ({wave_pol.upper()}, backing={backing}) | Uncertainty: {unc_state}\n"
            f"Points: {len(sweep)}\n"
            f"Metal loss dB mean/min/max: {metal_mean:.3f} / {metal_min:.3f} / {metal_max:.3f}\n"
            f"Air loss dB mean/min/max: {air_mean:.3f} / {air_min:.3f} / {air_max:.3f}\n"
            f"Insertion loss dB mean/min/max: {ins_mean:.3f} / {ins_min:.3f} / {ins_max:.3f}\n"
            f"Metal phase deg mean/min/max: {phase_mean:.3f} / {phase_min:.3f} / {phase_max:.3f}\n"
            f"Contiguous bandwidth with metal loss >= 10 dB at 0 deg: {bw10:.3f} GHz"
        )

    def _selected_idx(self) -> int | None:
        sel = self.layer_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _on_layer_preview_configure(self, _event: tk.Event) -> None:
        self._draw_layer_preview()

    def _draw_layer_preview(self) -> None:
        self.layer_preview.delete("all")
        colors = self._colors

        width = max(self.layer_preview.winfo_width(), int(self.layer_preview.cget("width")))
        height = max(self.layer_preview.winfo_height(), int(self.layer_preview.cget("height")))
        if width < 40 or height < 40:
            return

        if not self.layers:
            self.layer_preview.create_text(
                width / 2.0,
                height / 2.0,
                text="No layers configured",
                fill=colors["preview_empty"],
            )
            return

        pad = 12.0
        title_gap = 18.0
        x0 = pad
        x1 = width - pad
        y0 = pad + title_gap
        y1 = height - pad - title_gap
        if y1 <= y0:
            return

        self.layer_preview.create_text(
            (x0 + x1) * 0.5,
            pad,
            text="Top (incident side)",
            anchor="n",
            fill=colors["preview_text"],
        )
        self.layer_preview.create_text(
            (x0 + x1) * 0.5,
            height - pad,
            text="Bottom / backing",
            anchor="s",
            fill=colors["preview_text"],
        )

        thicknesses = [max(layer.thickness_in, 0.0) for layer in self.layers]
        n = len(self.layers)
        stack_h = y1 - y0
        min_h = min(22.0, stack_h / max(float(n), 1.0))
        min_total = min_h * n

        if stack_h <= min_total or sum(thicknesses) <= 0:
            heights = [stack_h / n] * n
        else:
            extra_h = stack_h - min_total
            total_t = sum(thicknesses)
            heights = [min_h + extra_h * (t / total_t) for t in thicknesses]

        layer_colors = colors["layer_colors"]

        y = y0
        for i, (layer, layer_h) in enumerate(zip(self.layers, heights), start=1):
            yn = y1 if i == n else y + layer_h
            fill = layer_colors[(i - 1) % len(layer_colors)]
            self.layer_preview.create_rectangle(
                x0,
                y,
                x1,
                yn,
                fill=fill,
                outline=colors["preview_layer_border"],
                width=1,
            )

            material_name = Path(layer.file_0deg).stem or Path(layer.file_0deg).name or "material"
            layer_type = "aniso" if layer.anisotropic else "iso"
            label = f"{i}. {material_name} | {layer.thickness_in:g} in | {layer_type}"
            max_chars = max(16, int((x1 - x0) / 6.7))
            if len(label) > max_chars:
                label = label[: max_chars - 3] + "..."
            self.layer_preview.create_text(
                (x0 + x1) * 0.5,
                (y + yn) * 0.5,
                text=label,
                fill=colors["preview_layer_text"],
            )
            y = yn

        self.layer_preview.create_rectangle(x0, y0, x1, y1, outline=colors["preview_outline"], width=1)

    def _refresh_layers(self) -> None:
        self.layer_list.delete(0, tk.END)
        for i, layer in enumerate(self.layers, start=1):
            file0 = Path(layer.file_0deg).name or layer.file_0deg
            if layer.anisotropic:
                file90 = Path(layer.file_90deg).name or layer.file_90deg
                desc = (
                    f"{i}. t={layer.thickness_in:g} in | aniso | pol={layer.polarization_deg:g} deg | "
                    f"0deg={file0} | 90deg={file90}"
                )
            else:
                desc = f"{i}. t={layer.thickness_in:g} in | iso | file={file0}"
            self.layer_list.insert(tk.END, desc)
        self._draw_layer_preview()

    def _add_layer(self) -> None:
        dlg = LayerDialog(self, presets=BUILTIN_MATERIAL_PRESETS)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.layers.append(dlg.result)
            self._refresh_layers()

    def _edit_layer(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            messagebox.showwarning("Layer", "Select a layer to edit.")
            return
        dlg = LayerDialog(self, self.layers[idx], presets=BUILTIN_MATERIAL_PRESETS)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.layers[idx] = dlg.result
            self._refresh_layers()
            self.layer_list.selection_set(idx)

    def _remove_layer(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            messagebox.showwarning("Layer", "Select a layer to remove.")
            return
        del self.layers[idx]
        self._refresh_layers()

    def _move_up(self) -> None:
        idx = self._selected_idx()
        if idx is None or idx == 0:
            return
        self.layers[idx - 1], self.layers[idx] = self.layers[idx], self.layers[idx - 1]
        self._refresh_layers()
        self.layer_list.selection_set(idx - 1)

    def _move_down(self) -> None:
        idx = self._selected_idx()
        if idx is None or idx >= len(self.layers) - 1:
            return
        self.layers[idx + 1], self.layers[idx] = self.layers[idx], self.layers[idx + 1]
        self._refresh_layers()
        self.layer_list.selection_set(idx + 1)

    def _load_layers(self, skiprows: int, layer_configs: list[LayerConfig] | None = None) -> list[LoadedLayer]:
        source_layers = self.layers if layer_configs is None else layer_configs
        loaded: list[LoadedLayer] = []
        for i, layer in enumerate(source_layers, start=1):
            t_m = layer.thickness_in * INCH_TO_M
            if t_m <= 0:
                raise ValueError(f"Layer {i}: thickness must be > 0.")

            table_0 = read_material_table(Path(layer.file_0deg), skiprows)
            table_90 = (
                read_material_table(Path(layer.file_90deg), skiprows)
                if layer.anisotropic
                else None
            )
            loaded.append(
                LoadedLayer(
                    thickness_m=t_m,
                    anisotropic=layer.anisotropic,
                    polarization_deg=layer.polarization_deg,
                    table_0deg=table_0,
                    table_90deg=table_90,
                )
            )
        return loaded

    def _set_dashboard_text(self, text: str) -> None:
        self.dashboard_text.configure(state="normal")
        self.dashboard_text.delete("1.0", tk.END)
        self.dashboard_text.insert("1.0", text)
        self.dashboard_text.configure(state="disabled")

    def _snapshot_layers(self) -> list[LayerConfig]:
        return [
            LayerConfig(
                thickness_in=layer.thickness_in,
                anisotropic=layer.anisotropic,
                file_0deg=layer.file_0deg,
                file_90deg=layer.file_90deg,
                polarization_deg=layer.polarization_deg,
            )
            for layer in self.layers
        ]

    def _set_task_state(self, running: bool, text: str) -> None:
        self._task_running = running
        state = "disabled" if running else "normal"
        for btn in (
            self.compute_btn,
            self.inv_run_btn,
            self.inv_apply_btn,
            self.layer_add_btn,
            self.layer_edit_btn,
            self.layer_remove_btn,
            self.layer_up_btn,
            self.layer_down_btn,
        ):
            if btn is not None:
                btn.configure(state=state)
        self.status_var.set(text)
        if self.status_progress is not None:
            if running:
                self.status_progress.grid()
                self.status_progress.start(10)
            else:
                self.status_progress.stop()
                self.status_progress.grid_remove()

    def _run_background_task(
        self,
        task_name: str,
        worker: Callable[[], _T],
        on_success: Callable[[_T], None],
        error_title: str,
    ) -> None:
        if self._task_running:
            messagebox.showwarning(task_name, "Another task is already running.")
            return

        result_q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        self._set_task_state(True, f"{task_name} running...")

        def _runner() -> None:
            try:
                result_q.put(("ok", worker()))
            except Exception as exc:
                result_q.put(("err", exc))

        threading.Thread(target=_runner, daemon=True).start()

        def _poll() -> None:
            try:
                status, payload = result_q.get_nowait()
            except queue.Empty:
                if self.winfo_exists():
                    self.after(100, _poll)
                return

            self._set_task_state(False, "Ready")
            if status == "ok":
                on_success(payload)  # type: ignore[arg-type]
            else:
                messagebox.showerror(error_title, str(payload))

        self.after(100, _poll)

    def _active_left_tab_label(self) -> str:
        if self.left_tabs is None:
            return ""
        try:
            selected = self.left_tabs.select()
            return str(self.left_tabs.tab(selected, "text"))
        except Exception:
            return ""

    def _is_angle_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Analysis"

    def _is_inverse_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Inverse Design"

    def _on_left_tab_changed(self, _event: object) -> None:
        self._update_plot()

    def _get_selected_metric_grid(self) -> tuple[str, str, list[list[float]]] | None:
        if self.last_heatmap_results is None:
            return None

        metric_label = self.heatmap_metric_var.get()
        metric_key = self.metric_label_to_key.get(metric_label)
        if metric_key is None:
            return None

        view_key = self.uncertainty_view_label_to_key.get(
            self.uncertainty_view_var.get(),
            "nominal",
        )
        base = self.last_heatmap_results[metric_key]
        if view_key == "nominal":
            return metric_label, metric_key, base

        if self.last_heatmap_uncertainty_min is None or self.last_heatmap_uncertainty_max is None:
            return metric_label, metric_key, base

        z_min = self.last_heatmap_uncertainty_min[metric_key]
        z_max = self.last_heatmap_uncertainty_max[metric_key]
        if view_key == "min":
            return f"{metric_label} [min]", metric_key, z_min
        if view_key == "max":
            return f"{metric_label} [max]", metric_key, z_max
        if NUMPY_AVAILABLE:
            span = (np.asarray(z_max, dtype=float) - np.asarray(z_min, dtype=float)).tolist()
        else:
            span = [
                [z_max[i][j] - z_min[i][j] for j in range(len(z_min[i]))]
                for i in range(len(z_min))
            ]
        return f"{metric_label} [span]", metric_key, span

    def _apply_manual_slices(self) -> None:
        if not self._is_angle_tab_active():
            messagebox.showwarning("Slice Input", "Switch to Analysis tab to edit heatmap slices.")
            return
        if self.last_heatmap_results is None:
            messagebox.showwarning("Slice Input", "Run heatmap mode first.")
            return

        angles = self.last_heatmap_results["angle_deg"]
        freqs = self.last_heatmap_results["freq_ghz"]
        angle_text = self.slice_angle_var.get().strip()
        freq_text = self.slice_freq_var.get().strip()
        if not angle_text and not freq_text:
            return

        try:
            if angle_text:
                angle_val = float(angle_text)
                if angle_val < angles[0] or angle_val > angles[-1]:
                    raise ValueError(f"Angle slice must be in [{angles[0]:g}, {angles[-1]:g}] deg.")
                self.selected_angle_idx = nearest_index(angles, angle_val)
            if freq_text:
                freq_val = float(freq_text)
                if freq_val < freqs[0] or freq_val > freqs[-1]:
                    raise ValueError(f"Frequency slice must be in [{freqs[0]:g}, {freqs[-1]:g}] GHz.")
                self.selected_freq_idx = nearest_index(freqs, freq_val)
        except Exception as exc:
            messagebox.showerror("Slice Input", str(exc))
            return

        self._update_plot()

    def _update_slice_plots(
        self,
        angles: list[float],
        freqs: list[float],
        z: list[list[float]],
        metric_label: str,
    ) -> None:
        colors = self._colors
        if (
            self.ax_freq_slice is None
            or self.ax_angle_slice is None
            or self.ax_heatmap is None
            or not angles
            or not freqs
        ):
            return

        if self.selected_angle_idx is None or self.selected_angle_idx >= len(angles):
            self.selected_angle_idx = len(angles) // 2
        if self.selected_freq_idx is None or self.selected_freq_idx >= len(freqs):
            self.selected_freq_idx = len(freqs) // 2

        j = self.selected_angle_idx
        i = self.selected_freq_idx
        angle_sel = angles[j]
        freq_sel = freqs[i]
        self.slice_angle_var.set(f"{angle_sel:.6g}")
        self.slice_freq_var.set(f"{freq_sel:.6g}")

        freq_slice = [row[j] for row in z]
        angle_slice = z[i]

        self.ax_freq_slice.clear()
        self.ax_freq_slice.plot(freqs, freq_slice, color=colors["plot_line_freq"], linewidth=1.7)
        self.ax_freq_slice.set_title(
            f"Frequency Slice @ {angle_sel:g} deg",
            fontsize=9,
            pad=2,
        )
        self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
        self.ax_freq_slice.set_ylabel(metric_label, fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        self.ax_angle_slice.clear()
        self.ax_angle_slice.plot(angles, angle_slice, color=colors["plot_line_angle"], linewidth=1.7)
        self.ax_angle_slice.set_title(
            f"Angle Slice @ {freq_sel:g} GHz",
            fontsize=9,
            pad=2,
        )
        self.ax_angle_slice.set_xlabel("Angle (deg)", fontsize=8)
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        if self.last_heatmap_uncertainty_min is not None and self.last_heatmap_uncertainty_max is not None:
            metric_key = self.metric_label_to_key[self.heatmap_metric_var.get()]
            zmin = self.last_heatmap_uncertainty_min[metric_key]
            zmax = self.last_heatmap_uncertainty_max[metric_key]
            self.ax_freq_slice.fill_between(
                freqs,
                [r[j] for r in zmin],
                [r[j] for r in zmax],
                color=colors["plot_line_freq"],
                alpha=0.12,
                linewidth=0,
            )
            self.ax_angle_slice.fill_between(
                angles,
                zmin[i],
                zmax[i],
                color=colors["plot_line_angle"],
                alpha=0.12,
                linewidth=0,
            )

        self.ax_heatmap.axvline(
            angle_sel,
            color=colors["plot_crosshair"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
        self.ax_heatmap.axhline(
            freq_sel,
            color=colors["plot_crosshair"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )

    def _on_plot_click(self, event: object) -> None:
        if (
            not MPL_AVAILABLE
            or self.ax_heatmap is None
            or self.last_heatmap_results is None
            or not self._is_angle_tab_active()
            or getattr(event, "inaxes", None) is not self.ax_heatmap
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return

        angles = self.last_heatmap_results["angle_deg"]
        freqs = self.last_heatmap_results["freq_ghz"]
        x = float(event.xdata)
        y = float(event.ydata)
        self.selected_angle_idx = nearest_index(angles, x)
        self.selected_freq_idx = nearest_index(freqs, y)
        self._update_plot()

    def _draw_inverse_placeholder(self, text: str) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        colors = self._colors
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Inverse Candidate Analysis")
        self.ax_heatmap.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color=colors["muted_text"],
        )
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        if self.ax_freq_slice is not None:
            self.ax_freq_slice.clear()
            self.ax_freq_slice.set_title("Score vs Total Thickness", fontsize=9, pad=2)
            self.ax_freq_slice.set_xlabel("Total thickness (in)", fontsize=8)
            self.ax_freq_slice.set_ylabel("Score (dB)", fontsize=8)
            self._style_plot_axis(self.ax_freq_slice)
            self.ax_freq_slice.grid(False)
        if self.ax_angle_slice is not None:
            self.ax_angle_slice.clear()
            self.ax_angle_slice.set_title("Robustness Gap", fontsize=9, pad=2)
            self.ax_angle_slice.set_xlabel("Candidate rank", fontsize=8)
            self.ax_angle_slice.set_ylabel("Nominal - Worst (dB)", fontsize=8)
            self._style_plot_axis(self.ax_angle_slice)
            self.ax_angle_slice.grid(False)
        self.canvas.draw_idle()

    def _update_inverse_plot(self) -> None:
        if (
            not MPL_AVAILABLE
            or self.ax_heatmap is None
            or self.ax_freq_slice is None
            or self.ax_angle_slice is None
            or self.canvas is None
        ):
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        if not self.inverse_candidates:
            self._draw_inverse_placeholder("Run inverse design to compare candidate stackups.")
            return

        colors = self._colors
        n = len(self.inverse_candidates)
        if (
            len(self.inverse_plot_samples) != n
            or not self.inverse_plot_freqs
        ):
            self._draw_inverse_placeholder("Run inverse design to compute percentile-vs-frequency curves.")
            return

        ranks = list(range(1, n + 1))
        scores = [c.score_db for c in self.inverse_candidates]
        worst = [c.worst_mean_db for c in self.inverse_candidates]
        nominal = [c.nominal_mean_db for c in self.inverse_candidates]
        total_thickness = [sum(c.thickness_in) for c in self.inverse_candidates]
        robustness_gap = [c.nominal_mean_db - c.worst_mean_db for c in self.inverse_candidates]

        selected_idx = 0
        if self.inv_results_list is not None:
            sel = self.inv_results_list.curselection()
            if sel:
                selected_idx = int(sel[0])
        selected_idx = max(0, min(selected_idx, n - 1))

        percentile = self._current_inverse_percentile()

        def _percentile(vals: list[float], p: float) -> float:
            if not vals:
                return float("nan")
            if NUMPY_AVAILABLE:
                return float(np.percentile(np.asarray(vals, dtype=float), p))
            sorted_vals = sorted(vals)
            if len(sorted_vals) == 1:
                return float(sorted_vals[0])
            pos = (p / 100.0) * (len(sorted_vals) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(sorted_vals[lo])
            t = pos - lo
            return float(sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t)

        curves: list[list[float]] = []
        for cand_samples in self.inverse_plot_samples:
            curve = [_percentile(freq_vals, percentile) for freq_vals in cand_samples]
            curves.append(curve)

        self.ax_heatmap.clear()
        freqs = self.inverse_plot_freqs
        for i, curve in enumerate(curves):
            if i == selected_idx:
                continue
            self.ax_heatmap.plot(freqs, curve, color=colors["plot_line_freq"], linewidth=1.0, alpha=0.25)
        self.ax_heatmap.plot(
            freqs,
            curves[selected_idx],
            color=colors["plot_line_angle"],
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=f"Selected candidate (P{percentile:g})",
        )
        self.ax_heatmap.set_title(f"Metal Loss vs Frequency at P{percentile:g} across analyzed points")
        self.ax_heatmap.set_xlabel("Frequency (GHz)")
        self.ax_heatmap.set_ylabel("Metal loss (dB)")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(True, color=colors["plot_grid"], alpha=0.3)
        self.ax_heatmap.legend(loc="best", fontsize=8)

        self.ax_freq_slice.clear()
        self.ax_freq_slice.scatter(total_thickness, scores, color=colors["plot_line_freq"], s=24, alpha=0.9)
        self.ax_freq_slice.scatter(
            [total_thickness[selected_idx]],
            [scores[selected_idx]],
            color=colors["plot_line_angle"],
            s=54,
            marker="*",
            zorder=3,
        )
        self.ax_freq_slice.set_title("Score vs Total Thickness", fontsize=9, pad=2)
        self.ax_freq_slice.set_xlabel("Total thickness (in)", fontsize=8)
        self.ax_freq_slice.set_ylabel("Score (dB)", fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        self.ax_angle_slice.clear()
        self.ax_angle_slice.bar(ranks, robustness_gap, color=colors["plot_line_angle"], alpha=0.75)
        self.ax_angle_slice.bar(
            [ranks[selected_idx]],
            [robustness_gap[selected_idx]],
            color=colors["accent"],
            alpha=0.95,
        )
        self.ax_angle_slice.set_title("Robustness Gap (Nominal - Worst)", fontsize=9, pad=2)
        self.ax_angle_slice.set_xlabel("Candidate rank", fontsize=8)
        self.ax_angle_slice.set_ylabel("Gap (dB)", fontsize=8)
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        cand = self.inverse_candidates[selected_idx]
        mat_names = ", ".join(Path(p).name for p in cand.material_files)
        if len(mat_names) > 48:
            mat_names = mat_names[:45] + "..."
        self.ax_angle_slice.text(
            0.02,
            0.98,
            f"#{selected_idx + 1} score={cand.score_db:.3f} dB\n"
            f"nom={cand.nominal_mean_db:.3f}, worst={cand.worst_mean_db:.3f}\n"
            f"materials: {mat_names}",
            transform=self.ax_angle_slice.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color=colors["text"],
        )
        self.canvas.draw_idle()

    def _draw_plot_placeholder(self, text: str) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        colors = self._colors
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Heatmap")
        self.ax_heatmap.set_xlabel("Angle (deg)")
        self.ax_heatmap.set_ylabel("Frequency (GHz)")
        self.ax_heatmap.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color=colors["muted_text"],
        )
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        if self.ax_freq_slice is not None:
            self.ax_freq_slice.clear()
            self.ax_freq_slice.set_title("Metric vs Frequency", fontsize=9, pad=2)
            self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
            self.ax_freq_slice.set_ylabel("Metric", fontsize=8)
            self._style_plot_axis(self.ax_freq_slice)
            self.ax_freq_slice.grid(False)
        if self.ax_angle_slice is not None:
            self.ax_angle_slice.clear()
            self.ax_angle_slice.set_title("Metric vs Angle", fontsize=9, pad=2)
            self.ax_angle_slice.set_xlabel("Angle (deg)", fontsize=8)
            self._style_plot_axis(self.ax_angle_slice)
            self.ax_angle_slice.grid(False)
        self.canvas.draw_idle()

    def _update_plot(self) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        if self.plot_frame is not None:
            if self._is_inverse_tab_active():
                self.plot_frame.configure(text="Inverse Candidate Plots")
                self._update_inverse_plot()
                return
            self.plot_frame.configure(text="Heatmap")
            if not self._is_angle_tab_active():
                self._draw_plot_placeholder("Heatmap and slice plots are shown on the Analysis tab.")
                return
        colors = self._colors
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None

        if self.last_heatmap_results is None:
            self._draw_plot_placeholder("Run heatmap mode to populate plot.")
            return

        selected = self._get_selected_metric_grid()
        if selected is None:
            self._draw_plot_placeholder("Select a valid metric.")
            return
        metric_label, metric_key, z = selected
        try:
            cmin, cmax = self._get_color_limits()
        except Exception as exc:
            messagebox.showerror("Colorbar", str(exc))
            return

        angles = self.last_heatmap_results["angle_deg"]
        freqs = self.last_heatmap_results["freq_ghz"]
        self.ax_heatmap.clear()
        cmap = "magma" if "[span]" in metric_label else ("twilight" if "phase" in metric_key else "viridis")
        im = self.ax_heatmap.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=(angles[0], angles[-1], freqs[0], freqs[-1]),
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
        )
        self.ax_heatmap.set_title(f"{metric_label} vs Angle/Frequency")
        self.ax_heatmap.set_xlabel("Angle (deg)")
        self.ax_heatmap.set_ylabel("Frequency (GHz)")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        self.heatmap_cbar = self.fig.colorbar(im, ax=self.ax_heatmap)
        if cmin is None or cmax is None:
            self.heatmap_cbar.set_label(metric_label)
        else:
            self.heatmap_cbar.set_label(f"{metric_label} [{cmin:g}, {cmax:g}]")
        style_colorbar(self.heatmap_cbar, colors)

        self._update_slice_plots(angles, freqs, z, metric_label)
        self.canvas.draw_idle()

    def _compute_heatmap_data(
        self,
        loaded_layers: list[LoadedLayer],
        wave_pol: str,
        angles: list[float],
        freqs: list[float] | None = None,
        thickness_scale: float = 1.0,
        eps_scale: float = 1.0,
        mu_scale: float = 1.0,
    ) -> dict[str, list[list[float]] | list[float]]:
        if freqs is None:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            freqs = make_sweep(f_start, f_stop, f_step)

        for i, layer in enumerate(loaded_layers, start=1):
            validate_sweep_coverage(freqs, layer.table_0deg, f"layer {i} 0 deg/isotropic")
            if layer.anisotropic:
                if layer.table_90deg is None:
                    raise ValueError(f"Layer {i}: anisotropic layer is missing a 90 deg table.")
                validate_sweep_coverage(freqs, layer.table_90deg, f"layer {i} 90 deg")

        if NUMPY_AVAILABLE:
            grids = {k: np.zeros((len(freqs), len(angles)), dtype=float) for k in HEATMAP_METRIC_KEYS}
            for j, a in enumerate(angles):
                col = compute_angle_metrics_many(
                    freqs,
                    a,
                    loaded_layers,
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                )
                for key in HEATMAP_METRIC_KEYS:
                    grids[key][:, j] = np.asarray(col[key], dtype=float)
            metric_grids = {k: grids[k].tolist() for k in HEATMAP_METRIC_KEYS}
        else:
            metric_grids = {k: [] for k in HEATMAP_METRIC_KEYS}
            for f_ghz in freqs:
                row = compute_angle_metrics(
                    f_ghz,
                    angles[0],
                    loaded_layers,
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                )
                for key in HEATMAP_METRIC_KEYS:
                    metric_grids[key].append([row[key]])
            for j in range(1, len(angles)):
                for i, f_ghz in enumerate(freqs):
                    m = compute_angle_metrics(
                        f_ghz,
                        angles[j],
                        loaded_layers,
                        wave_pol,
                        thickness_scale=thickness_scale,
                        eps_scale=eps_scale,
                        mu_scale=mu_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        metric_grids[key][i].append(m[key])

        return {
            "angle_deg": angles,
            "freq_ghz": freqs,
            **metric_grids,
        }

    def _compute_frequency_mode(
        self,
        output_path: Path,
        include_header: bool,
        loaded_layers: list[LoadedLayer],
        backing: str,
        uncertainty: UncertaintyConfig,
        sweep: list[float],
        wave_pol: str,
    ) -> tuple[int, str]:
        for i, layer in enumerate(loaded_layers, start=1):
            validate_sweep_coverage(sweep, layer.table_0deg, f"layer {i} 0 deg/isotropic")
            if layer.anisotropic:
                if layer.table_90deg is None:
                    raise ValueError(f"Layer {i}: anisotropic layer is missing a 90 deg table.")
                validate_sweep_coverage(sweep, layer.table_90deg, f"layer {i} 90 deg")

        z_nom = compute_stack_impedance_many(sweep, loaded_layers, backing)
        scales = build_uncertainty_scales(uncertainty)
        envelope_enabled = uncertainty.enabled and len(scales) > 1

        if not envelope_enabled:
            rows = [(f_ghz, z.real, z.imag) for f_ghz, z in zip(sweep, z_nom)]
            write_output(output_path, rows, include_header)
        else:
            zr_nom = [z.real for z in z_nom]
            zi_nom = [z.imag for z in z_nom]
            zr_min = zr_nom.copy()
            zr_max = zr_nom.copy()
            zi_min = zi_nom.copy()
            zi_max = zi_nom.copy()
            for t_scale, e_scale, m_scale in scales:
                if is_nominal_scale(t_scale, e_scale, m_scale):
                    continue
                z_s = compute_stack_impedance_many(
                    sweep,
                    loaded_layers,
                    backing,
                    thickness_scale=t_scale,
                    eps_scale=e_scale,
                    mu_scale=m_scale,
                )
                for i, z in enumerate(z_s):
                    zr = z.real
                    zi = z.imag
                    zr_min[i] = min(zr_min[i], zr)
                    zr_max[i] = max(zr_max[i], zr)
                    zi_min[i] = min(zi_min[i], zi)
                    zi_max[i] = max(zi_max[i], zi)

            with output_path.open("w", encoding="utf-8") as f:
                if include_header:
                    f.write(
                        "frequency_GHz z_r z_i z_r_min z_r_max z_i_min z_i_max\n"
                    )
                for i, f_ghz in enumerate(sweep):
                    f.write(
                        f"{f_ghz:.12g} {zr_nom[i]:.12g} {zi_nom[i]:.12g} "
                        f"{zr_min[i]:.12g} {zr_max[i]:.12g} {zi_min[i]:.12g} {zi_max[i]:.12g}\n"
                    )

        summary = self._summarize_frequency_run(
            sweep,
            loaded_layers,
            wave_pol,
            envelope_enabled,
            backing,
        )
        return len(sweep), summary

    def _compute_angle_mode(
        self,
        output_path: Path,
        include_header: bool,
        loaded_layers: list[LoadedLayer],
        uncertainty: UncertaintyConfig,
        angles: list[float],
        freqs: list[float],
        wave_pol: str,
    ) -> tuple[
        int,
        dict[str, list[list[float]] | list[float]],
        dict[str, list[list[float]]] | None,
        dict[str, list[list[float]]] | None,
        str,
    ]:
        out = self._compute_heatmap_data(loaded_layers, wave_pol, angles, freqs=freqs)

        scales = build_uncertainty_scales(uncertainty)
        envelope_enabled = uncertainty.enabled and len(scales) > 1
        envelope_min: dict[str, list[list[float]]] | None = None
        envelope_max: dict[str, list[list[float]]] | None = None
        if envelope_enabled:
            if NUMPY_AVAILABLE:
                envelope_min = {
                    key: np.asarray(out[key], dtype=float)
                    for key in HEATMAP_METRIC_KEYS
                }
                envelope_max = {
                    key: np.asarray(out[key], dtype=float)
                    for key in HEATMAP_METRIC_KEYS
                }
                for t_scale, e_scale, m_scale in scales:
                    if is_nominal_scale(t_scale, e_scale, m_scale):
                        continue
                    s_out = self._compute_heatmap_data(
                        loaded_layers,
                        wave_pol,
                        angles,
                        freqs=freqs,
                        thickness_scale=t_scale,
                        eps_scale=e_scale,
                        mu_scale=m_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        arr = np.asarray(s_out[key], dtype=float)
                        envelope_min[key] = np.minimum(envelope_min[key], arr)
                        envelope_max[key] = np.maximum(envelope_max[key], arr)
                envelope_min = {key: envelope_min[key].tolist() for key in HEATMAP_METRIC_KEYS}
                envelope_max = {key: envelope_max[key].tolist() for key in HEATMAP_METRIC_KEYS}
            else:
                envelope_min = {
                    key: [[v for v in row] for row in out[key]]
                    for key in HEATMAP_METRIC_KEYS
                }
                envelope_max = {
                    key: [[v for v in row] for row in out[key]]
                    for key in HEATMAP_METRIC_KEYS
                }
                for t_scale, e_scale, m_scale in scales:
                    if is_nominal_scale(t_scale, e_scale, m_scale):
                        continue
                    s_out = self._compute_heatmap_data(
                        loaded_layers,
                        wave_pol,
                        angles,
                        freqs=freqs,
                        thickness_scale=t_scale,
                        eps_scale=e_scale,
                        mu_scale=m_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        for i in range(len(out["freq_ghz"])):
                            for j in range(len(out["angle_deg"])):
                                val = s_out[key][i][j]
                                envelope_min[key][i][j] = min(envelope_min[key][i][j], val)
                                envelope_max[key][i][j] = max(envelope_max[key][i][j], val)

        freq = out["freq_ghz"]
        ang = out["angle_deg"]
        metal_loss = out["metal_loss_db"]
        metal_phase = out["metal_phase_deg"]
        air_loss = out["air_loss_db"]
        air_phase = out["air_phase_deg"]
        insertion_loss = out["insertion_loss_db"]
        insertion_phase = out["insertion_phase_deg"]

        with output_path.open("w", encoding="utf-8") as f:
            if include_header:
                if envelope_enabled:
                    f.write(
                        "frequency_GHz angle_deg "
                        "metal_loss_db metal_loss_db_min metal_loss_db_max "
                        "metal_phase_deg metal_phase_deg_min metal_phase_deg_max "
                        "air_loss_db air_loss_db_min air_loss_db_max "
                        "air_phase_deg air_phase_deg_min air_phase_deg_max "
                        "insertion_loss_db insertion_loss_db_min insertion_loss_db_max "
                        "insertion_phase_deg insertion_phase_deg_min insertion_phase_deg_max\n"
                    )
                else:
                    f.write(
                        "frequency_GHz angle_deg metal_loss_db metal_phase_deg "
                        "air_loss_db air_phase_deg insertion_loss_db insertion_phase_deg\n"
                    )
            for i, f_ghz in enumerate(freq):
                for j, a in enumerate(ang):
                    if envelope_enabled:
                        if envelope_min is None or envelope_max is None:
                            raise ValueError("Internal error: uncertainty envelopes are unavailable.")
                        f.write(
                            f"{f_ghz:.12g} {a:.12g} "
                            f"{metal_loss[i][j]:.12g} "
                            f"{envelope_min['metal_loss_db'][i][j]:.12g} {envelope_max['metal_loss_db'][i][j]:.12g} "
                            f"{metal_phase[i][j]:.12g} "
                            f"{envelope_min['metal_phase_deg'][i][j]:.12g} {envelope_max['metal_phase_deg'][i][j]:.12g} "
                            f"{air_loss[i][j]:.12g} "
                            f"{envelope_min['air_loss_db'][i][j]:.12g} {envelope_max['air_loss_db'][i][j]:.12g} "
                            f"{air_phase[i][j]:.12g} "
                            f"{envelope_min['air_phase_deg'][i][j]:.12g} {envelope_max['air_phase_deg'][i][j]:.12g} "
                            f"{insertion_loss[i][j]:.12g} "
                            f"{envelope_min['insertion_loss_db'][i][j]:.12g} {envelope_max['insertion_loss_db'][i][j]:.12g} "
                            f"{insertion_phase[i][j]:.12g} "
                            f"{envelope_min['insertion_phase_deg'][i][j]:.12g} {envelope_max['insertion_phase_deg'][i][j]:.12g}\n"
                        )
                    else:
                        f.write(
                            f"{f_ghz:.12g} {a:.12g} "
                            f"{metal_loss[i][j]:.12g} {metal_phase[i][j]:.12g} "
                            f"{air_loss[i][j]:.12g} {air_phase[i][j]:.12g} "
                            f"{insertion_loss[i][j]:.12g} {insertion_phase[i][j]:.12g}\n"
                        )

        summary = self._summarize_angle_run(out, wave_pol, envelope_enabled)
        return len(freq) * len(ang), out, envelope_min, envelope_max, summary

    def _thickness_candidates(self, t_min: float, t_max: float, t_steps: int) -> list[float]:
        if t_min <= 0 or t_max <= 0:
            raise ValueError("Optimization thickness bounds must be > 0.")
        if t_max < t_min:
            raise ValueError("Optimization thickness max must be >= min.")
        if t_steps <= 0:
            raise ValueError("Optimization thickness steps must be >= 1.")
        if t_steps == 1:
            return [t_min]
        return [t_min + i * (t_max - t_min) / (t_steps - 1) for i in range(t_steps)]

    def _discover_material_candidates(self, skiprows: int) -> list[str]:
        out: list[str] = []
        for p in sorted(Path(".").glob("*.txt")):
            try:
                read_material_table(p, skiprows)
            except Exception:
                continue
            out.append(str(p))
        return out

    def _read_inverse_uncertainty_config(self) -> UncertaintyConfig:
        if not self.inv_uncertainty_var.get():
            return UncertaintyConfig(enabled=False, thickness_pct=0.0, eps_pct=0.0, mu_pct=0.0)

        t_pct = float(self.inv_unc_t_pct_var.get().strip())
        eps_pct = float(self.inv_unc_eps_pct_var.get().strip())
        mu_pct = float(self.inv_unc_mu_pct_var.get().strip())
        if t_pct < 0 or eps_pct < 0 or mu_pct < 0:
            raise ValueError("Inverse-design uncertainty percentages must be >= 0.")
        return UncertaintyConfig(enabled=True, thickness_pct=t_pct, eps_pct=eps_pct, mu_pct=mu_pct)

    def _parse_inverse_discrete_freqs(self, text: str) -> list[float]:
        tokens = (
            text.replace(",", " ")
            .replace(";", " ")
            .replace("\n", " ")
            .split()
        )
        if not tokens:
            raise ValueError("Enter one or more discrete frequencies in GHz (for example: 8.2, 9.5, 10.0).")
        values: list[float] = []
        for token in tokens:
            values.append(float(token))
        unique_sorted = sorted(set(values))
        if not unique_sorted:
            raise ValueError("No valid discrete frequencies were provided.")
        return unique_sorted

    def _score_inverse_candidate(
        self,
        target_freqs: list[float],
        target_angles: list[float],
        candidate_layers: list[LoadedLayer],
        wave_pol: str,
        scales: list[tuple[float, float, float]],
        score_mode: str,
    ) -> tuple[float, float, float, float, float]:
        corner_means: list[float] = []
        nominal_mean: float | None = None
        for t_scale, e_scale, m_scale in scales:
            values: list[float] = []
            for angle_deg in target_angles:
                metrics = compute_angle_metrics_many(
                    target_freqs,
                    angle_deg,
                    candidate_layers,
                    wave_pol,
                    thickness_scale=t_scale,
                    eps_scale=e_scale,
                    mu_scale=m_scale,
                )
                values.extend(metrics["metal_loss_db"])
            mean_db, _mn, _mx = self._stats(values)
            corner_means.append(mean_db)
            if abs(t_scale - 1.0) < 1e-12 and abs(e_scale - 1.0) < 1e-12 and abs(m_scale - 1.0) < 1e-12:
                nominal_mean = mean_db

        if not corner_means:
            raise ValueError("No corner scores computed for inverse candidate.")
        if nominal_mean is None:
            nominal_mean = corner_means[0]

        worst_mean = min(corner_means)
        avg_mean = sum(corner_means) / len(corner_means)
        best_mean = max(corner_means)
        score_db = worst_mean if "worst-case" in score_mode.lower() else avg_mean
        return score_db, nominal_mean, worst_mean, avg_mean, best_mean

    def _refresh_inverse_results_list(self) -> None:
        if self.inv_results_list is None:
            return
        self.inv_results_list.delete(0, tk.END)
        for i, c in enumerate(self.inverse_candidates, start=1):
            t_text = ", ".join(f"{t:.4g}" for t in c.thickness_in)
            m_text = ", ".join(Path(p).name for p in c.material_files)
            line = (
                f"{i:02d}: score={c.score_db:.3f} dB | nom={c.nominal_mean_db:.3f} | "
                f"worst={c.worst_mean_db:.3f} | avg={c.avg_mean_db:.3f} | "
                f"t=[{t_text}] | m=[{m_text}]"
            )
            self.inv_results_list.insert(tk.END, line)
        self._update_plot()

    def _apply_inverse_candidate(self) -> None:
        try:
            if not self.inverse_candidates or self.inv_results_list is None:
                messagebox.showwarning("Inverse Design", "Run inverse design first.")
                return
            sel = self.inv_results_list.curselection()
            if not sel:
                messagebox.showwarning("Inverse Design", "Select a candidate to apply.")
                return
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.inverse_candidates):
                messagebox.showwarning("Inverse Design", "Selected candidate is out of range.")
                return

            cand = self.inverse_candidates[idx]
            if len(cand.thickness_in) != len(self.layers) or len(cand.material_files) != len(self.layers):
                raise ValueError("Layer count changed since inverse design run. Re-run inverse design.")

            for i, layer in enumerate(self.layers):
                layer.thickness_in = cand.thickness_in[i]
                if not layer.anisotropic:
                    layer.file_0deg = cand.material_files[i]
            self._refresh_layers()

            msg = (
                f"Applied inverse candidate #{idx + 1}.\n"
                f"Score: {cand.score_db:.3f} dB | Nominal: {cand.nominal_mean_db:.3f} dB | "
                f"Worst-case: {cand.worst_mean_db:.3f} dB"
            )
            self._set_dashboard_text(msg)
            messagebox.showinfo("Inverse Design", msg)
        except Exception as exc:
            messagebox.showerror("Inverse Design Error", str(exc))

    def _run_inverse_design(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer before inverse design.")

            layer_snapshot = self._snapshot_layers()
            skiprows = int(self.skiprows_var.get().strip())
            wave_pol = normalize_wave_polarization(self.inv_wave_pol_var.get())
            freq_mode = self.inv_freq_mode_var.get().strip().lower()
            if freq_mode.startswith("discrete"):
                target_freqs = self._parse_inverse_discrete_freqs(self.inv_freq_list_var.get())
                target_freq_desc = "Discrete GHz: " + ", ".join(f"{v:g}" for v in target_freqs)
            else:
                f_start = float(self.inv_target_start_var.get().strip())
                f_stop = float(self.inv_target_stop_var.get().strip())
                f_step = float(self.inv_target_step_var.get().strip())
                target_freqs = make_sweep(f_start, f_stop, f_step)
                target_freq_desc = f"Band GHz: {f_start:g}-{f_stop:g} (step {f_step:g})"
            a_start = float(self.inv_angle_start_var.get().strip())
            a_stop = float(self.inv_angle_stop_var.get().strip())
            if a_start < 0 or a_stop > 90:
                raise ValueError("Inverse-design angle range must satisfy 0 <= start and stop <= 90 deg.")
            if abs(a_stop - a_start) <= 1e-12:
                target_angles = [a_start]
            else:
                a_step = float(self.inv_angle_step_var.get().strip())
                target_angles = make_sweep(a_start, a_stop, a_step)

            t_min = float(self.inv_thick_min_var.get().strip())
            t_max = float(self.inv_thick_max_var.get().strip())
            t_steps = int(self.inv_thick_steps_var.get().strip())
            max_evals = int(self.inv_max_evals_var.get().strip())
            top_n = int(self.inv_top_n_var.get().strip())
            if max_evals <= 0:
                raise ValueError("Inverse-design Max evals must be >= 1.")
            if top_n <= 0:
                raise ValueError("Inverse-design Top N must be >= 1.")
            score_mode = self.inv_score_mode_var.get().strip()
            material_mode = self.inv_material_mode_var.get().strip().lower()
            uncertainty_cfg = self._read_inverse_uncertainty_config()
        except Exception as exc:
            messagebox.showerror("Inverse Design Error", str(exc))
            return

        def worker() -> tuple[list[InverseCandidate], str, list[float], list[list[list[float]]]]:
            thickness_grid = self._thickness_candidates(t_min, t_max, t_steps)
            scales = build_uncertainty_scales(uncertainty_cfg)
            search_materials = "search" in material_mode
            material_pool = self._discover_material_candidates(skiprows) if search_materials else []
            if search_materials and not material_pool:
                raise ValueError("No valid material tables (*.txt) found for inverse-design material search mode.")

            layer_material_options: list[list[str]] = []
            for layer in layer_snapshot:
                if layer.anisotropic:
                    layer_material_options.append([layer.file_0deg])
                elif search_materials:
                    layer_material_options.append(material_pool)
                else:
                    layer_material_options.append([layer.file_0deg])

            thickness_combo_count = len(thickness_grid) ** len(layer_snapshot)
            material_combo_count = 1
            for opts in layer_material_options:
                material_combo_count *= max(1, len(opts))
            total_combo_count = thickness_combo_count * material_combo_count

            table_cache: dict[str, MaterialTable] = {}

            def get_table(path_str: str) -> MaterialTable:
                key = str(Path(path_str))
                if key not in table_cache:
                    table_cache[key] = read_material_table(Path(key), skiprows)
                return table_cache[key]

            def build_candidate_layers(cand: InverseCandidate) -> list[LoadedLayer]:
                layers_out: list[LoadedLayer] = []
                for i, layer in enumerate(layer_snapshot):
                    table_0 = get_table(cand.material_files[i])
                    table_90: MaterialTable | None = None
                    if layer.anisotropic:
                        table_90 = get_table(layer.file_90deg)
                    layers_out.append(
                        LoadedLayer(
                            thickness_m=cand.thickness_in[i] * INCH_TO_M,
                            anisotropic=layer.anisotropic,
                            polarization_deg=layer.polarization_deg,
                            table_0deg=table_0,
                            table_90deg=table_90,
                        )
                    )
                return layers_out

            top_candidates: list[InverseCandidate] = []
            eval_count = 0
            skipped_out_of_range = 0
            stop = False

            material_index_ranges = [range(len(opts)) for opts in layer_material_options]
            for mat_indices in product(*material_index_ranges):
                chosen_files = [layer_material_options[i][idx] for i, idx in enumerate(mat_indices)]
                candidate_tables_0: list[MaterialTable] = []
                candidate_tables_90: list[MaterialTable | None] = []
                coverage_ok = True
                for i, layer in enumerate(layer_snapshot, start=1):
                    table_0 = get_table(chosen_files[i - 1])
                    try:
                        validate_sweep_coverage(target_freqs, table_0, f"inverse layer {i} 0deg/isotropic")
                    except Exception:
                        coverage_ok = False
                        break
                    table_90: MaterialTable | None = None
                    if layer.anisotropic:
                        table_90 = get_table(layer.file_90deg)
                        try:
                            validate_sweep_coverage(target_freqs, table_90, f"inverse layer {i} 90deg")
                        except Exception:
                            coverage_ok = False
                            break
                    candidate_tables_0.append(table_0)
                    candidate_tables_90.append(table_90)
                if not coverage_ok:
                    skipped_out_of_range += len(thickness_grid) ** len(layer_snapshot)
                    continue

                for thicknesses in product(thickness_grid, repeat=len(layer_snapshot)):
                    eval_count += 1
                    candidate_layers: list[LoadedLayer] = []
                    for i, layer in enumerate(layer_snapshot):
                        candidate_layers.append(
                            LoadedLayer(
                                thickness_m=thicknesses[i] * INCH_TO_M,
                                anisotropic=layer.anisotropic,
                                polarization_deg=layer.polarization_deg,
                                table_0deg=candidate_tables_0[i],
                                table_90deg=candidate_tables_90[i],
                            )
                        )

                    score_db, nominal_mean, worst_mean, avg_mean, best_mean = self._score_inverse_candidate(
                        target_freqs,
                        target_angles,
                        candidate_layers,
                        wave_pol,
                        scales,
                        score_mode,
                    )
                    top_candidates.append(
                        InverseCandidate(
                            score_db=score_db,
                            nominal_mean_db=nominal_mean,
                            worst_mean_db=worst_mean,
                            avg_mean_db=avg_mean,
                            best_mean_db=best_mean,
                            thickness_in=[float(v) for v in thicknesses],
                            material_files=chosen_files[:],
                        )
                    )
                    top_candidates.sort(key=lambda c: c.score_db, reverse=True)
                    if len(top_candidates) > top_n:
                        top_candidates.pop()

                    if eval_count >= max_evals:
                        stop = True
                        break
                if stop:
                    break

            if not top_candidates:
                raise ValueError("Inverse design found no valid candidates in the target region.")

            inverse_samples: list[list[list[float]]] = []
            for cand in top_candidates:
                cand_layers = build_candidate_layers(cand)
                freq_samples = [[] for _ in target_freqs]
                for t_scale, e_scale, m_scale in scales:
                    for angle_deg in target_angles:
                        metrics = compute_angle_metrics_many(
                            target_freqs,
                            angle_deg,
                            cand_layers,
                            wave_pol,
                            thickness_scale=t_scale,
                            eps_scale=e_scale,
                            mu_scale=m_scale,
                        )
                        for fi, val in enumerate(metrics["metal_loss_db"]):
                            freq_samples[fi].append(val)
                inverse_samples.append(freq_samples)

            best = top_candidates[0]
            unc_state = "enabled" if uncertainty_cfg.enabled else "disabled"
            msg = (
                f"Inverse design complete.\n"
                f"Objective: {score_mode}\n"
                f"Region: {target_freq_desc}, {a_start:g}-{a_stop:g} deg, pol={wave_pol.upper()}\n"
                f"Uncertainty: {unc_state} ({len(scales)} corner(s))\n"
                f"Evaluated: {eval_count} candidates (limit {max_evals}, theoretical {total_combo_count})\n"
                f"Skipped out-of-range combos (estimated): {skipped_out_of_range}\n"
                f"Best score: {best.score_db:.3f} dB | nominal {best.nominal_mean_db:.3f} dB | "
                f"worst-case {best.worst_mean_db:.3f} dB\n"
                f"Stored top {len(top_candidates)} candidates. Use Apply Selected to update the stack."
            )
            return top_candidates, msg, [float(v) for v in target_freqs], inverse_samples

        def on_success(result: tuple[list[InverseCandidate], str, list[float], list[list[list[float]]]]) -> None:
            self.inverse_candidates, msg, freqs_plot, samples_plot = result
            self.inverse_plot_freqs = freqs_plot
            self.inverse_plot_samples = samples_plot
            self._refresh_inverse_results_list()
            if self.inv_results_list is not None:
                self.inv_results_list.selection_clear(0, tk.END)
                self.inv_results_list.selection_set(0)
            self._set_dashboard_text(msg)
            self._update_plot()
            messagebox.showinfo("Inverse Design", msg)

        self._run_background_task("Inverse Design", worker, on_success, "Inverse Design Error")

    def _compute(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")

            layer_snapshot = self._snapshot_layers()
            angle_mode = self.angle_mode_var.get()
            skiprows = int(self.skiprows_var.get().strip())
            output_path = Path(self.output_var.get().strip())
            include_header = self.header_var.get()
            uncertainty = self._read_uncertainty_config()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            freqs = make_sweep(f_start, f_stop, f_step)
            wave_pol = normalize_wave_polarization(self.wave_pol_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if angle_mode:
            try:
                a_start = float(self.angle_start_var.get().strip())
                a_stop = float(self.angle_stop_var.get().strip())
                if a_start < 0 or a_stop > 90:
                    raise ValueError("Angle range must satisfy 0 <= start and stop <= 90 deg.")
                if abs(a_stop - a_start) <= 1e-12:
                    angles = [a_start]
                else:
                    a_step = float(self.angle_step_var.get().strip())
                    angles = make_sweep(a_start, a_stop, a_step)
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
                return

            def worker() -> dict[str, object]:
                loaded_layers = self._load_layers(skiprows, layer_snapshot)
                n, out, env_min, env_max, summary = self._compute_angle_mode(
                    output_path,
                    include_header,
                    loaded_layers,
                    uncertainty,
                    angles,
                    freqs,
                    wave_pol,
                )
                return {
                    "mode": "angle",
                    "count": n,
                    "summary": summary,
                    "out": out,
                    "env_min": env_min,
                    "env_max": env_max,
                }

            def on_success(result: dict[str, object]) -> None:
                self.last_heatmap_results = result["out"]  # type: ignore[assignment]
                self.last_heatmap_uncertainty_min = result["env_min"]  # type: ignore[assignment]
                self.last_heatmap_uncertainty_max = result["env_max"]  # type: ignore[assignment]
                self.selected_angle_idx = None
                self.selected_freq_idx = None
                self._set_dashboard_text(str(result["summary"]))
                self._update_plot()
                messagebox.showinfo("Complete", f"Wrote {int(result['count'])} heatmap points to:\n{output_path}")

            self._run_background_task("Compute", worker, on_success, "Error")
            return

        try:
            backing = normalize_backing(self.backing_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(skiprows, layer_snapshot)
            n, summary = self._compute_frequency_mode(
                output_path,
                include_header,
                loaded_layers,
                backing,
                uncertainty,
                freqs,
                wave_pol,
            )
            return {"mode": "freq", "count": n, "summary": summary}

        def on_success(result: dict[str, object]) -> None:
            self.last_heatmap_results = None
            self.last_heatmap_uncertainty_min = None
            self.last_heatmap_uncertainty_max = None
            self.selected_angle_idx = None
            self.selected_freq_idx = None
            self._set_dashboard_text(str(result["summary"]))
            self._update_plot()
            messagebox.showinfo("Complete", f"Wrote {int(result['count'])} frequency points to:\n{output_path}")

        self._run_background_task("Compute", worker, on_success, "Error")


def main() -> None:
    if not TK_AVAILABLE:
        raise SystemExit(
            "Tkinter is not available. Install Tk support for Python to run the GUI."
        )
    app = ImpedanceGui()
    app.mainloop()


if __name__ == "__main__":
    main()
