import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import traceback
import os
from pathlib import Path
import datetime
from .main import rerun

class PyScreenRerunApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("PyScreen Rerun Suite")
        self.geometry("800x600")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.results_files = []

        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_container, 
            text="PyScreen Rerun Analysis", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=(0, 20))

        # Description
        self.description_label = ctk.CTkLabel(
            self.main_container,
            text="Select results files from PyScreen main analysis to rerun specific peaks\nmarked in the 'rerun' column using external NIST search.",
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        self.description_label.pack(pady=(0, 20))

        # Results Files Selection Section
        self.files_frame = ctk.CTkFrame(self.main_container)
        self.files_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # File selection header
        self.files_header = ctk.CTkLabel(
            self.files_frame, 
            text="Results Files Selection:", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.files_header.pack(anchor="w", padx=10, pady=(10, 5))

        # File list frame with grid layout
        self.file_list_frame = ctk.CTkFrame(self.files_frame)
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Configure grid weights for responsive layout
        self.file_list_frame.grid_columnconfigure(1, weight=1)
        self.file_list_frame.grid_rowconfigure(0, weight=1)

        # Label for listbox
        self.listbox_label = ctk.CTkLabel(self.file_list_frame, text="Selected Results Files:")
        self.listbox_label.grid(row=0, column=0, sticky="nw", padx=5, pady=2)

        # Listbox for showing selected files
        self.results_listbox = tk.Listbox(self.file_list_frame, height=8, width=60)
        self.results_listbox.grid(row=0, column=1, sticky="nsew", padx=5, pady=2)

        # Scrollbar for listbox
        self.scrollbar = tk.Scrollbar(self.file_list_frame, orient="vertical")
        self.scrollbar.grid(row=0, column=2, sticky="ns", pady=2)
        self.results_listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.results_listbox.yview)

        # Buttons frame
        self.buttons_frame = ctk.CTkFrame(self.file_list_frame)
        self.buttons_frame.grid(row=0, column=3, sticky="ns", padx=5, pady=2)

        self.add_files_btn = ctk.CTkButton(
            self.buttons_frame, 
            text="Add Files", 
            command=self._browse_results_files, 
            width=120
        )
        self.add_files_btn.pack(fill="x", pady=2)

        self.remove_selected_btn = ctk.CTkButton(
            self.buttons_frame, 
            text="Remove Selected", 
            command=self._remove_selected_file, 
            width=120
        )
        self.remove_selected_btn.pack(fill="x", pady=2)

        self.clear_all_btn = ctk.CTkButton(
            self.buttons_frame, 
            text="Clear All", 
            command=self._clear_all_files, 
            width=120
        )
        self.clear_all_btn.pack(fill="x", pady=2)

        # Info section
        self.info_frame = ctk.CTkFrame(self.main_container)
        self.info_frame.pack(fill="x", padx=10, pady=5)

        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="ℹ️ This tool will process results files and rerun NIST search for peaks marked in the 'rerun' column.\nCorresponding data files (.txt) must be in the same directory as the results files.",
            font=ctk.CTkFont(size=11),
            justify="left",
            text_color="gray"
        )
        self.info_label.pack(padx=10, pady=10, anchor="w")

        # Progress Bar and Status
        self.status_frame = ctk.CTkFrame(self.main_container)
        self.status_frame.pack(fill="x", padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", width=100)
        self.status_label.pack(side="left", padx=5, pady=5)

        # Run Button
        self.run_button = ctk.CTkButton(
            self.main_container, 
            text="Run Rerun Analysis", 
            command=self.run_rerun_analysis,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.run_button.pack(pady=15, padx=10, fill="x")

        # Error display area (initially hidden)
        self.error_details_visible = False
        self.error_frame = ctk.CTkFrame(self.main_container)
        self.error_label = ctk.CTkLabel(
            self.error_frame, 
            text="An error occurred. Click to expand details.", 
            text_color="orange"
        )
        self.error_label.pack(fill="x")
        self.error_label.bind("<Button-1>", self.toggle_error_details)

    def _browse_results_files(self):
        """Browse and select multiple results files"""
        filetypes = (
            ("Excel files", "*.xlsx"),
            ("Excel files", "*.xls"), 
            ("All files", "*.*")
        )
        initialdir = str(Path.home())
        
        files = filedialog.askopenfilenames(
            title="Select Results Files",
            filetypes=filetypes,
            initialdir=initialdir,
            parent=self
        )
        
        if files:
            for f in files:
                if f not in self.results_files:
                    # Validate that it looks like a results file
                    if "_results" in Path(f).stem or f.endswith(('.xlsx', '.xls')):
                        self.results_files.append(f)
                    else:
                        messagebox.showwarning(
                            "File Warning", 
                            f"File {Path(f).name} doesn't appear to be a results file.\nAdding anyway, but please ensure it's correct."
                        )
                        self.results_files.append(f)
            self._update_files_display()

    def _remove_selected_file(self):
        """Remove selected file from the list"""
        selected_indices = list(self.results_listbox.curselection())
        for i in reversed(selected_indices):
            del self.results_files[i]
        self._update_files_display()

    def _clear_all_files(self):
        """Clear all files from the list"""
        self.results_files.clear()
        self._update_files_display()

    def _update_files_display(self):
        """Update the listbox display with current files"""
        self.results_listbox.delete(0, tk.END)
        for f_path in self.results_files:
            self.results_listbox.insert(tk.END, Path(f_path).name)

    def run_rerun_analysis(self):
        """Start the rerun analysis"""
        if not self.results_files:
            messagebox.showerror("Input Error", "Please select at least one results file.")
            return

        # Validate files exist
        missing_files = [f for f in self.results_files if not Path(f).exists()]
        if missing_files:
            messagebox.showerror(
                "File Error", 
                f"The following files do not exist:\n" + "\n".join([Path(f).name for f in missing_files])
            )
            return

        self.run_button.configure(state="disabled", text="Running Rerun Analysis...")
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.status_label.configure(text="Processing...")
        self.hide_error_details()

        # Run in a separate thread to keep GUI responsive
        thread = threading.Thread(target=self._run_rerun_thread, args=(list(self.results_files),))
        thread.daemon = True
        thread.start()

    def _run_rerun_thread(self, results_paths):
        """Run the rerun analysis in a separate thread"""
        try:
            # Call the rerun function from main.py
            rerun(results_paths)
            self.after(0, self.on_rerun_complete)
        except Exception as e:
            tb_str = traceback.format_exc()
            # Save error to file automatically
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = f"pyscreen_rerun_errors_{timestamp}.txt"
            filepath = os.path.join(os.getcwd(), filename)
            try:
                with open(filepath, "w") as f:
                    f.write(f"PyScreen Rerun Error:\n{str(e)}\n\nFull traceback:\n{tb_str}")
            except Exception as file_err:
                print(f"Failed to write error log: {file_err}")
            
            self.after(0, self.on_rerun_error, e, tb_str)

    def on_rerun_complete(self):
        """Handle successful completion of rerun analysis"""
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.progress_bar.configure(mode="determinate")
        self.status_label.configure(text="Rerun Analysis Completed Successfully!")
        self.run_button.configure(state="normal", text="Run Rerun Analysis")
        
        messagebox.showinfo(
            "Success", 
            "PyScreen rerun analysis completed successfully!"
        )

    def on_rerun_error(self, error_exception, traceback_str):
        """Handle error during rerun analysis"""
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="determinate")
        self.status_label.configure(text="Error Occurred!")
        self.run_button.configure(state="normal", text="Run Rerun Analysis")
        self.show_error_popup("Rerun Analysis Error", f"{str(error_exception)}\n\nFull traceback:\n{traceback_str}")

    def show_error_popup(self, title, message):
        """Show error message with expandable details"""
        self.error_label.configure(text=f"{title}. Click to expand details.")
        
        if not self.error_frame.winfo_ismapped():
            self.error_frame.pack(fill="x", padx=10, pady=5, before=self.status_frame)
        
        # Store the error message for the details window
        self._last_error_message = message

    def toggle_error_details(self, event=None):
        """Show error details in a separate window"""
        if hasattr(self, 'error_details_window') and self.error_details_window.winfo_exists():
            self.error_details_window.lift()
            return

        # Create a new resizable Toplevel window for error details
        self.error_details_window = tk.Toplevel(self)
        self.error_details_window.title("Error Details")
        self.error_details_window.geometry("700x400")
        self.error_details_window.resizable(True, True)

        # Text area for error message
        text_area = tk.Text(self.error_details_window, wrap="word")
        text_area.insert("1.0", getattr(self, "_last_error_message", "No error message."))
        text_area.configure(state="disabled")
        text_area.pack(fill="both", expand=True, padx=10, pady=10)

        # Button frame
        button_frame = ctk.CTkFrame(self.error_details_window)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Export button
        export_btn = ctk.CTkButton(
            button_frame, 
            text="Export Error Message", 
            command=lambda: self.export_error_message(getattr(self, "_last_error_message", ""))
        )
        export_btn.pack(side="left", padx=(0, 10), pady=5)

        # Close button
        close_btn = ctk.CTkButton(button_frame, text="Close", command=self.error_details_window.destroy)
        close_btn.pack(side="right", pady=5)

        self.error_details_window.focus_set()

    def export_error_message(self, message):
        """Export error message to a file"""
        now = datetime.datetime.now()
        filename = f"pyscreen_rerun_error_{now.strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Error Message",
            initialfilename=filename
        )
        
        if filepath:
            try:
                with open(filepath, "w") as f:
                    f.write(f"PyScreen Rerun Error Report\n")
                    f.write(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*50 + "\n\n")
                    f.write(message)
                messagebox.showinfo("Export Successful", f"Error message exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not export error message:\n{e}")

    def hide_error_details(self):
        """Hide error details frame"""
        if self.error_frame.winfo_ismapped():
            self.error_frame.pack_forget()


def main():
    """Main function to run the rerun GUI"""
    app = PyScreenRerunApp()
    app.mainloop()


if __name__ == "__main__":
    main()