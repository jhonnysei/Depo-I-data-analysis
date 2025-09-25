# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:35:58 2025

@author: Johannes Seibel
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from io import BytesIO
from PIL import Image
import win32clipboard
import os
import io
import glob
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, medfilt


# --- Custom toolbar with Copy button ---
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent):
                
        # Do not pack inside NavigationToolbar2Tk
        super().__init__(canvas, parent, pack_toolbar=False)
        
        # Add custom copy button
        self.copy_btn = tk.Button(self, text="Copy Image", command=self.copy_image)
        self.copy_btn.pack(side=tk.LEFT, padx=2, pady=2)

    def copy_image(self):
        # Copy the current figure to clipboard
        fig = self.canvas.figure
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        image = Image.open(buf)

        output = BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # BMP header removed
        output.close()

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        print("Figure copied to clipboard!")
        

# --- Helper to display figures ---
def display_figure(fig, window_title):
    plot_window = tk.Toplevel()
    plot_window.title(window_title)
    plot_window.maxsize(1600, 1300)
    plot_window.minsize(900, 700)

    canvas_frame = tk.Frame(plot_window)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

     # Canvas
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Toolbar at the top
    toolbar = CustomToolbar(canvas, canvas_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)
    toolbar.zoom()

        
        # --- Main app ---
class DataAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depo-I MSTDS")
        self.root.geometry("1000x500")  # compact window
        self.data_frames = []

         # --- File Selection Mode (Standard vs Legacy Files) ---
        self.file_mode = tk.StringVar(value="standard")  # default mode

        file_mode_frame = tk.Frame(root)
        file_mode_frame.pack(pady=5)

        tk.Label(file_mode_frame, text="File Selection Mode:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        tk.Radiobutton(
            file_mode_frame, text="Standard", variable=self.file_mode,
            value="standard", font=("Arial", 12)
        ).grid(row=0, column=1, padx=5)

        tk.Radiobutton(
            file_mode_frame, text="Legacy Files", variable=self.file_mode,
            value="legacy", font=("Arial", 12)
        ).grid(row=0, column=2, padx=5)


        # Open file button
        tk.Button(root, text="Open File", command=self.open_file, width=20, height=2).pack(pady=5)

        # Buttons frame
        self.graph_buttons_frame = tk.Frame(root)
        self.graph_buttons_frame.pack(pady=10)

        # Graph buttons
        buttons_info_row1 = [
            ("Raw Heatmap", self.plot_raw_contour),
            ("Interpolated Heatmap", self.plot_interpolated_contour),
            ("Log Scale", self.plot_log_contour),
            ("Extract MS", self.extract_and_plot_MS),
            ("Extract Thermal Profile", self.extract_temp_dist)
        ]
        
        for i, (text, cmd) in enumerate(buttons_info_row1):
            b = tk.Button(self.graph_buttons_frame, text=text, width=20, height=2, command=cmd, state=tk.DISABLED)
            b.grid(row=1, column=i, padx=5, pady=5)
        
        # Graph buttons (row 2)
        buttons_info_row2 = [
            ("Plot Single MS", self.plot_single_MS),
            ("Plot Multiple MS", self.plot_multiple_MS)
        ]
        
        for i, (text, cmd) in enumerate(buttons_info_row2):
            b = tk.Button(self.graph_buttons_frame, text=text, width=20, height=2, command=cmd, state=tk.DISABLED)
            b.grid(row=2, column=i, padx=5, pady=5)    
        
        self.graph_buttons = self.graph_buttons_frame.winfo_children()
        
        # --- Interpolation Method Row ---
        self.interp_frame = tk.Frame(root)
        self.interp_frame.pack(pady=5)

        tk.Label(self.interp_frame, text="Interpolation Method:", font=("Arial", 12)).grid(row=0, column=0, padx=5)

        self.interp_method = tk.StringVar(value="linear")  # default method

        methods = [("Linear", "linear"), ("Cubic", "cubic"), ("Gaussian", "gaussian")]
        for i, (label, value) in enumerate(methods):
            tk.Radiobutton(
                self.interp_frame,
                text=label,
                variable=self.interp_method,
                value=value,
                font=("Arial", 12)
            ).grid(row=0, column=i+1, padx=5)
     
        # Gaussian sigma input field
        tk.Label(self.interp_frame, text="σ:", font=("Arial", 12)).grid(row=0, column=len(methods)+1, padx=5)
        self.gaussian_sigma_entry = tk.Entry(self.interp_frame, width=5, font=("Arial", 12))
        self.gaussian_sigma_entry.grid(row=0, column=len(methods)+2, padx=5)
        self.gaussian_sigma_entry.insert(0, "1")  # default sigma value
         
        # --- Mass Offset Row ---
        self.offset_frame = tk.Frame(root)
        self.offset_frame.pack(pady=5)
        
        # Entry field for offset value
        tk.Label(self.offset_frame, text="Mass Offset:", font=("Arial", 12)).grid(row=2, column=0, padx=5)
        self.mass_offset_entry = tk.Entry(self.offset_frame, width=10, font=("Arial", 12))
        self.mass_offset_entry.grid(row=2, column=1, padx=5)
        self.mass_offset_entry.insert(0, "0")  # Default offset value
        
        # Button to apply the offset
        self.offset_button = tk.Button(
            self.offset_frame,
            text="Apply Mass Offset",
            font=("Arial", 12),
            width=20,
            height=2,
            command=self.apply_mass_offset
        )
        self.offset_button.grid(row=2, column=2, padx=5)
       
        # --- Multi-file Search Row ---
        self.search_frame = tk.Frame(root)
        self.search_frame.pack(pady=5)
        
        # Labels + entries
        tk.Label(self.search_frame, text="Mass:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        self.mass_entry = tk.Entry(self.search_frame, width=10, font=("Arial", 12))
        self.mass_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(self.search_frame, text="Tolerance:", font=("Arial", 12)).grid(row=0, column=2, padx=5)
        self.tolerance_entry = tk.Entry(self.search_frame, width=10, font=("Arial", 12))
        self.tolerance_entry.grid(row=0, column=3, padx=5)
        
        # Buttons
        self.search_button = tk.Button(
            self.search_frame,
            text="Search for Mass",
            font=("Arial", 12),
            width=20,
            height=2,
            state=tk.DISABLED,   # initially disabled
            command=self.search_for_mass  # placeholder
        )
        self.search_button.grid(row=0, column=4, padx=5)
        
        self.create_plots_button = tk.Button(
            self.search_frame,
            text="Create Plots",
            font=("Arial", 12),
            width=20,
            height=2,
            state=tk.DISABLED,   # initially disabled
            command=self.create_plots  # placeholder
        )
        self.create_plots_button.grid(row=0, column=5, padx=5)

        # In __init__ after creating other controls
        self.show_original = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Show Original", variable=self.show_original).pack(pady=2)

    # --- File loading ---
    
    
    def _parse_counts_file(self, counts_path):
        """Return the counts Series (2nd column) parsed from a CASE2 counts file."""
       
        # Read as cp1252 (Windows), tolerate odd bytes
        with open(counts_path, "r", encoding="cp1252", errors="replace") as f:
            lines = f.readlines()
    
        # Normalize: decimal commas -> dots; semicolons -> tabs (delimiter)
        cleaned = []
        for ln in lines:
            ln = ln.replace(",", ".").replace(";", "\t")
            cleaned.append(ln)
    
        buf = io.StringIO("".join(cleaned))
        # Split on tabs or spaces; ignore comment lines
        counts_df = pd.read_csv(buf, sep=r"[\t ]+", header=None, engine="python", comment="#")
        # Drop completely empty columns if any
        counts_df = counts_df.dropna(axis=1, how="all")
    
        if counts_df.shape[1] < 2:
            raise ValueError(f"Counts file '{counts_path}' must have at least 2 columns, got {counts_df.shape[1]}.")
    
        counts = pd.to_numeric(counts_df.iloc[:, 1], errors="coerce")
        if counts.isna().all():
            raise ValueError(f"Second column in counts file '{counts_path}' could not be parsed as numeric.")
    
        return counts.reset_index(drop=True)

    
    def _load_case2_counts(self, main_path, prefix_len=12):
        """Find and load the counts Series for a CASE2 main file based on filename prefix."""
        folder = os.path.dirname(main_path)
        prefix = os.path.basename(main_path)[:prefix_len]
    
        # Try common case variants of .Asc
        patterns = [f"{prefix}*.Asc", f"{prefix}*.ASC", f"{prefix}*.asc"]
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(folder, pat)))
    
        # Exclude the main file itself (in case it also matches)
        main_abs = os.path.abspath(main_path)
        candidates = [p for p in candidates if os.path.abspath(p) != main_abs]
    
        if not candidates:
            raise FileNotFoundError(
                f"No counts file found in '{folder}' with prefix '{prefix}' (looking for {patterns})."
            )
    
        last_err = None
        for path in candidates:
            try:
                counts = self._parse_counts_file(path)
                return counts, path
            except Exception as e:
                last_err = e
    
        raise ValueError(
            f"Found {len(candidates)} candidate counts file(s) for prefix '{prefix}', "
            f"but none could be parsed. Last error: {last_err}"
        )
    
    
    def ask_open_noext_files(self):
        """Custom dialog to select files without dot in their name."""
        root = tk.Tk()
        root.withdraw()  # hide main window
    
        # First choose directory
        folder = filedialog.askdirectory(title="Select Folder with Files (no extension)")
        if not folder:
            return []
    
        # Collect only files without a dot in the name
        files = [f for f in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, f)) and "." not in f]
    
        if not files:
            messagebox.showinfo("No Files", "No files without extension found in this folder.")
            return []
    
        # Create selection window
        sel_win = tk.Toplevel()
        sel_win.title("Select Files Without Extension")
    
        listbox = tk.Listbox(sel_win, selectmode=tk.MULTIPLE, width=60, height=20)
        listbox.pack(padx=10, pady=10)
    
        for f in files:
            listbox.insert(tk.END, f)
    
        selected_files = []
    
        def confirm():
            indices = listbox.curselection()
            for i in indices:
                selected_files.append(os.path.join(folder, files[i]))
            sel_win.destroy()
    
        def select_all():
            listbox.select_set(0, tk.END)  # highlight all entries

        btn_frame = tk.Frame(sel_win)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Select All", command=select_all).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="OK", command=confirm).grid(row=0, column=1, padx=5)

        sel_win.wait_window()
        return selected_files

    
    def open_file(self, mode="noext"):
           
        """Open file(s) based on selected mode."""
        if self.file_mode.get() == "standard":
            file_paths = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=[("All Files", "*.*"), ("Text Files", "*.txt"), ("Asc Files", "*.Asc")]
            )
        else:  # Legacy mode → only files without extension
            file_paths = self.ask_open_noext_files()
        
        # return file_paths
        
        if not file_paths:
            messagebox.showinfo("No File Selected", "Please select at least one file.")
            return
    
    
    
        expected_columns = [
            "Time", "Vout", "Mass", "Trigger", "AI0 Ion",
            "AI3 Mass", "AI5 Thermo", "AI7", "Counts"
        ]
    
        self.data_frames.clear()
    
        for file_path in file_paths:
            try:
                # Quick sniff for CASE2
                with open(file_path, "r", encoding="cp1252", errors="replace") as f:
                    first_line = f.readline()
    
                if "Messprotokoll" in first_line:
                    # ---- CASE2: data starts at row 8; ignore headers; drop last col; assign names
                    df = pd.read_csv(
                        file_path,
                        sep="\t",
                        header=None,
                        skiprows=8,            # data lines only
                        encoding="cp1252",
                        engine="python"
                    )
    
                    # Keep first 10 cols, then drop the 10th (Signal)
                    if df.shape[1] < 10:
                        raise ValueError(f"CASE2 file '{os.path.basename(file_path)}' has {df.shape[1]} columns; expected ≥ 10.")
                    df = df.iloc[:, :10].copy()
                    df = df.drop(columns=df.columns[9])
    
                    # Map to standard names (order-based)
                    df.columns = expected_columns
    
                    # Attach counts from companion file (must exist; else raise)
                    counts, counts_path = self._load_case2_counts(file_path, prefix_len=12)
    
                    # Normalize commas in main df before numeric conversion
                    for col in df.columns:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    
                    # Convert main df to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
    
                    # Validate row counts
                    if len(counts) != len(df):
                        min_len = min(len(counts), len(df))
                        df = df.iloc[:min_len].reset_index(drop=True)
                        counts = counts.iloc[:min_len].reset_index(drop=True)
    
                    # Overwrite Counts column with the loaded counts
                    df["Counts"] = counts.values
    
                else:
                    # ---- CASE1: read as-is and normalize names
                    df = pd.read_csv(file_path, sep="\t", header=0, encoding="cp1252", engine="python")
                    # Enforce standard columns (by name if present, else by position)
                    if set(expected_columns).issubset(df.columns):
                        df = df[expected_columns].copy()
                    else:
                        df = df.iloc[:, :9].copy()
                        df.columns = expected_columns
    
                    # Normalize commas, then numeric
                    for col in df.columns:
                        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
    
                # Apply AI5 Thermo scaling (both cases)
                if "AI5 Thermo" in df.columns:
                    df["AI5 Thermo"] = df["AI5 Thermo"] * 250 + 300
    
                self.data_frames.append((file_path, df))
                print(f"Loaded: {file_path}")
    
            except Exception as e:
                messagebox.showerror("Error", f"Could not load {file_path}:\n{e}")
    
        if self.data_frames:
            if len(self.data_frames) == 1:
                # Enable single-file graph buttons
                for btn in self.graph_buttons:
                    btn.config(state=tk.NORMAL)
                # Disable multi-file buttons
                self.search_button.config(state=tk.DISABLED)
                self.create_plots_button.config(state=tk.DISABLED)
            else:
                # Disable single-file graph buttons
                for btn in self.graph_buttons:
                    btn.config(state=tk.DISABLED)
                # Enable multi-file buttons
                self.search_button.config(state=tk.NORMAL)
                self.create_plots_button.config(state=tk.NORMAL)

        print("File(s) loaded.")
        print(df)

        
    # Mass Offset
    def apply_mass_offset(self):
        """Apply the mass offset entered in the field to all loaded dataframes."""
        try:
            offset = float(self.mass_offset_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric offset.")
            return
    
        if not self.data_frames:
            messagebox.showwarning("No Data", "Please load at least one file first.")
            return
    
        # Apply the offset to all loaded dataframes
        for i, (file_path, df) in enumerate(self.data_frames):
            if 'Mass' in df.columns:
                df['Mass'] = df['Mass'] + offset
                self.data_frames[i] = (file_path, df)  # Update the tuple with modified df
    
        messagebox.showinfo("Mass Offset Applied", f"A mass offset of {offset} has been applied.")


    def _prepare_numeric_data(self, df, columns):
        """Ensure specified columns are numeric and drop duplicates for 'Mass' and 'AI5 Thermo'."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.drop_duplicates(subset=['Mass', 'AI5 Thermo'])
        return df    


    def interpolate_data(self, df, method="linear", n_points=500):
        """Interpolate Counts onto a regular Mass grid."""
        df = self._prepare_numeric_data(df, ["Mass", "Counts"])
        if df.empty:
            return None, None
    
        masses = df["Mass"].values
        counts = df["Counts"].values
    
        # Regular mass grid
        m_grid = np.linspace(masses.min(), masses.max(), n_points)
    
        # Interpolated counts
        i_grid = griddata(masses, counts, m_grid, method=method, fill_value=0)
    
        return m_grid, i_grid


    def plot_raw_contour(self):
        _, df = self.data_frames[-1]
        df = self._prepare_numeric_data(df, ["Mass", "AI5 Thermo", "Counts"])
        x_unique = np.sort(df["Mass"].unique())
        y_unique = np.sort(df["AI5 Thermo"].unique())
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = df.pivot(index="AI5 Thermo", columns="Mass", values="Counts").fillna(0)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        levels = np.linspace(Z.min().min(), Z.max().max(), 100)
        c = ax.contourf(X, Y, Z.values, levels=levels, cmap="viridis")
        fig.colorbar(c, ax=ax, label="Counts")
        ax.set_xlabel("Mass")
        ax.set_ylabel("AI5 Thermo")
        ax.set_title("Raw Heatmap")
        display_figure(fig, "Raw Heatmap")

    def plot_interpolated_contour(self):
        _, df = self.data_frames[-1]
        df = self._prepare_numeric_data(df, ["Mass", "AI5 Thermo", "Counts"])
        x = df["Mass"]
        y = df["AI5 Thermo"]
        z = df["Counts"]
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)
        
        method = self.interp_method.get()

        if method in ["linear", "cubic"]:
            # Use scipy griddata for linear or cubic
            zi = griddata((x, y), z, (xi, yi), method=method, fill_value=0)
        elif method == "gaussian":
            # First do linear interpolation
            zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=0)
            # Apply Gaussian smoothing
            try:
                sigma = float(self.gaussian_sigma_entry.get())
            except ValueError:
                sigma = 1
            zi = gaussian_filter(zi, sigma=sigma)
        else:
            messagebox.showerror("Error", f"Unknown interpolation method: {method}")
            return
        
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        c = ax.contourf(xi, yi, zi, levels=100, cmap="viridis")
        fig.colorbar(c, ax=ax, label="Counts")
        ax.set_xlabel("m/z", fontsize = 14)
        ax.set_ylabel("Temperature /K", fontsize = 14)
        ax.set_title(" ")
        display_figure(fig, " ")

    def plot_log_contour(self):
        _, df = self.data_frames[-1]
        df = self._prepare_numeric_data(df, ["Mass", "AI5 Thermo", "Counts"])
        x = df["Mass"]
        y = df["AI5 Thermo"]
        z = df["Counts"].replace(0, 1)  # Avoid log(0)
        
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)
        
        method = self.interp_method.get()

        if method in ["linear", "cubic"]:
            # Use scipy griddata for linear or cubic
            zi = griddata((x, y), z, (xi, yi), method=method, fill_value=0)
        elif method == "gaussian":
            # First do linear interpolation
            zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=0)
            # Apply Gaussian smoothing
            zi = gaussian_filter(zi, sigma=1)  # <-- you can make sigma user-adjustable
        else:
            messagebox.showerror("Error", f"Unknown interpolation method: {method}")
            return
        
        zi = np.clip(zi, 1, None)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        c = ax.contourf(xi, yi, zi, levels=np.logspace(0, np.log10(zi.max()), 100),
                        cmap="viridis", norm=mcolors.LogNorm(vmin=1, vmax=zi.max()))
        # Colorbar with nice log ticks
        cb = fig.colorbar(c, ax=ax, label="Counts (log scale)")
        cb.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None, numticks=10))
        cb.ax.yaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=False))
        
        ax.set_xlabel("m/z", fontsize = 14)
        ax.set_ylabel("Temperature /K", fontsize = 14)
        ax.set_title(" ")
        display_figure(fig, " ")

    def plot_single_MS(self):
        _, df = self.data_frames[-1]
        df = self._prepare_numeric_data(df, ["Mass", "AI0 Ion"])
        df["AI0 Ion"] = -df["AI0 Ion"]  # Multiply by -1
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(df["Mass"], df["AI0 Ion"])
        ax.set_xlabel("m/z", fontsize = 14)
        ax.set_ylabel("Intensity /a.u.", fontsize = 14)
        ax.set_title(" ")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        display_figure(fig, "Mass Spectrum")

    def plot_multiple_MS(self):
        pass

    def extract_and_plot_MS(self):
        if len(self.data_frames) != 1:
            messagebox.showwarning("Selection Error", "Please load exactly one file.")
            return
    
        _, df = self.data_frames[0]
    
        masses, counts = self.extract_single_spectrum(df)
        if masses is None:
            messagebox.showerror("Error", "Could not extract spectrum.")
            return
    
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(masses, counts)
        ax.set_xlabel("m/z", fontsize=14)
        ax.set_ylabel("Intensity / a.u.", fontsize=14)
        ax.set_title("Extracted Spectrum")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        display_figure(fig, "Extracted Spectrum")

    def extract_temp_dist(self):
        if len(self.data_frames) != 1:
            messagebox.showwarning("Selection Error", "Please load exactly one file.")
            return
    
        _, df = self.data_frames[0]
    
        temps, counts, xi, ys = self.extract_temperature_profile(df)
        if temps is None:
            messagebox.showerror("Error", "Could not extract spectrum.")
            return
    
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
    
        # Toggle original display
        if self.show_original.get():
            ax.plot(temps, counts, color="gray", alpha=0.6, label="Original")
    
        # Plot smoothed data (highlighted)
        ax.plot(xi, ys, color="blue", linewidth=2, label="Thermal Profile")
    
        ax.set_xlabel("Temperature /K", fontsize=14)
        ax.set_ylabel("Intensity / a.u.", fontsize=14)
        ax.set_title("Extracted Thermal Profile")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.legend()
    
        display_figure(fig, "Extracted Spectrum")

    def extract_single_spectrum(self, df):
        """
        Build a 1D spectrum by summing counts across all temperatures for each mass.
        """
        df = self._prepare_numeric_data(df, ["Mass", "Counts"])
        if df.empty:
            return None, None
    
        # Sum counts for each unique mass
        spec = (
            df.groupby("Mass", as_index=True)["Counts"]
            .sum()
            .sort_index()
        )
    
        masses = spec.index.values.astype(float)
        counts = spec.values.astype(float)
    
        return masses, counts
  
    def extract_temperature_profile(self, df,
                                    method='gaussian',
                                    median_kernel=3,
                                    window_fraction=0.02,
                                    savgol_polyorder=3,
                                    resample_points=1200):
        """
        Build a 1D thermal spectrum by summing counts across all masses
        for each temperature (AI5 Thermo), with optional smoothing.
    
        Returns
        -------
        temps : np.ndarray
            Original temperatures.
        counts : np.ndarray
            Original intensities.
        xi : np.ndarray
            Resampled temperature axis (uniform grid).
        ys : np.ndarray
            Smoothed intensities.
        """
        df = self._prepare_numeric_data(df, ["AI5 Thermo", "Counts"])
        if df.empty:
            return None, None, None, None
    
        # Sum counts for each unique temperature
        profile = (
            df.groupby("AI5 Thermo", as_index=True)["Counts"]
            .sum()
            .sort_index()
        )
    
        temps = profile.index.values.astype(float)
        counts = profile.values.astype(float)
    
        def _to_uniform_grid(x, y, num=1000, kind='linear'):
            """Interpolate x,y onto a uniform grid of length num. Returns xi, yi."""
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.size == 0:
                return np.array([]), np.array([])
        
            # sort by x
            order = np.argsort(x)
            x_s = x[order]
            y_s = y[order]
        
            # collapse duplicate x by averaging
            uniq_x, idx = np.unique(x_s, return_index=True)
            if uniq_x.size != x_s.size:
                from collections import defaultdict
                acc = defaultdict(list)
                for xx, yy in zip(x_s, y_s):
                    acc[xx].append(yy)
                uniq_x = np.array(sorted(acc.keys()))
                y_s = np.array([np.mean(acc[k]) for k in uniq_x])
                x_s = uniq_x
        
            xi = np.linspace(x_s.min(), x_s.max(), num)
            f = interp1d(x_s, y_s, kind=kind, bounds_error=False, fill_value=0.0)
            yi = f(xi)
            return xi, yi
        
        # --- smoothing ---
        xi, yi = _to_uniform_grid(temps, counts, num=resample_points, kind='linear')
    
        # Median filter (optional)
        if method in ('median', 'median+savgol'):
            k = int(median_kernel)
            if k % 2 == 0:
                k += 1
            if k < 3:
                k = 3
            yi = medfilt(yi, kernel_size=k)
    
        if method == 'savgol' or method == 'median+savgol':
            N = yi.size
            w = int(max(7, round(N * window_fraction)))
            if w % 2 == 0:
                w += 1
            if w >= N:
                w = N - 1
                if w % 2 == 0:
                    w -= 1
            if w < 3:
                w = 3
            po = min(savgol_polyorder, w - 1)
            ys = savgol_filter(yi, window_length=w, polyorder=po, mode='interp')
        elif method == 'gaussian':
            try:
                sigma = float(self.gaussian_sigma_entry.get())
            except ValueError:
                sigma = 1
            ys = gaussian_filter1d(yi, sigma=sigma)
        else:
            ys = yi.copy()
    
        # --- Normalize smoothed to same max as original ---
        if ys.max() > 0 and counts.max() > 0:
            ys = ys * (counts.max() / ys.max())
    
        return temps, counts, xi, ys

    def search_for_mass(self):
        if len(self.data_frames) < 2:
            messagebox.showwarning("Search Error", "Please load at least two files to search.")
            return
    
        try:
            target = float(self.mass_entry.get())
            tol = float(self.tolerance_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for Mass and Tolerance.")
            return
    
        lower, upper = target - tol, target + tol
        matching_files = []
    
        for file_path, df in self.data_frames:
            df = self._prepare_numeric_data(df, ["Mass", "Counts", "AI5 Thermo"])
            if df.empty:
                continue
            df = df.dropna(subset=["Mass", "Counts"])
            if df.empty:
                continue
    
            # --- NEW: Extract the "summed spectrum" ---
            masses, intens = self.extract_single_spectrum(df)
            if masses is None or intens is None or len(masses) == 0:
                continue
    
            # Restrict to tolerance window
            mask = (masses >= lower) & (masses <= upper)
            if not mask.any():
                continue
            masses = masses[mask]
            intens = intens[mask]
    
            if len(masses) < 3:
                continue
    
            # Interpolate onto uniform grid
            m_grid = np.linspace(masses.min(), masses.max(), max(100, len(masses)))
            i_grid = np.interp(m_grid, masses, intens)
    
            # Smooth
            if len(i_grid) >= 7:
                win = int(max(7, (len(i_grid) // 15) * 2 + 1))
                i_smooth = savgol_filter(i_grid, window_length=win, polyorder=2)
            else:
                i_smooth = gaussian_filter1d(i_grid, sigma=5)
    
            max_i = float(np.nanmax(i_smooth)) if np.isfinite(i_smooth).any() else 0.0
            if max_i <= 0:
                continue
    
            prominence = 0.5 * max_i
            height = 0.5 * max_i
            peaks, _ = find_peaks(i_smooth, prominence=prominence, height=height)
    
            # --- Debug plot ---
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(m_grid, i_grid, color="gray", alpha=0.5, label="Data")
            ax.plot(m_grid, i_smooth, color="blue", label="Smoothed")
            if len(peaks) > 0:
                ax.plot(m_grid[peaks], i_smooth[peaks], "ro", label="Peaks")
            ax.axvline(target, color="green", linestyle="--", label="Target mass")
            ax.axvspan(lower, upper, color="yellow", alpha=0.2, label="Tolerance window")
            ax.set_title(os.path.basename(file_path))
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.show()
    
            if len(peaks) > 0:
                matching_files.append(os.path.basename(file_path))
    
        if matching_files:
            messagebox.showinfo(
                "Search Results",
                f"Files with peaks near {target} ± {tol} m/z:\n" + "\n".join(matching_files)
            )
        else:
            messagebox.showinfo(
                "Search Results",
                f"No peaks found near {target} ± {tol} m/z."
            )

    def create_plots(self):
        if not self.data_frames:
            messagebox.showwarning("Warning", "No files loaded.")
            return
    
        # Ask for output folder
        output_dir = filedialog.askdirectory(title="Select Folder to Save Plots")
        if not output_dir:
            return  # user canceled
    
        method = self.interp_method.get()
        sigma = float(self.gaussian_sigma_entry.get()) if method == "Gaussian" else None

    
        for file_path, df in self.data_frames:
            try:
                # Prepare numeric data
                df = self._prepare_numeric_data(df, ["Mass", "AI5 Thermo", "Counts"])
                x = df["Mass"]
                y = df["AI5 Thermo"]
                z = df["Counts"]
    
                # Interpolation grid
                xi = np.linspace(x.min(), x.max(), 200)
                yi = np.linspace(y.min(), y.max(), 200)
                xi, yi = np.meshgrid(xi, yi)
                method = self.interp_method.get()

                if method in ["linear", "cubic"]:
                    # Use scipy griddata for linear or cubic
                    zi = griddata((x, y), z, (xi, yi), method=method, fill_value=0)
                elif method == "gaussian":
                    # First do linear interpolation
                    zi = griddata((x, y), z, (xi, yi), method="linear", fill_value=0)
                    # Apply Gaussian smoothing
                    try:
                        sigma = float(self.gaussian_sigma_entry.get())
                    except ValueError:
                        sigma = 1
                    zi = gaussian_filter(zi, sigma=sigma)
                else:
                    messagebox.showerror("Error", f"Unknown interpolation method: {method}")
                    return
    
                # Save with same base filename
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(output_dir, base_name + ".png")
                
                # Create figure
                fig = Figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                c = ax.contourf(xi, yi, zi, levels=100, cmap="viridis")
                fig.colorbar(c, ax=ax, label="Counts")
    
                ax.set_xlabel("m/z", fontsize=14)
                ax.set_ylabel("Temperature /K", fontsize=14)
                ax.set_title(base_name, fontsize=14)
    
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
    
            except Exception as e:
                messagebox.showerror("Error", f"Could not create plot for {file_path}: {e}")
                continue
    
        messagebox.showinfo("Success", f"Plots saved to:\n{output_dir}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()
